%% Data Parsing

csvMain = importdata('summary.csv');

% Initializing global variables
EatingAction = zeros(46080,5000);
NonEatingAction = zeros(44928,5000);
startPosEat = 1;
startPosNonEat = 1;
MaxColEating = 0;
MaxColNonEating = 0;
resultEatMat = zeros(2640,18,5);
resultNonEatMat = zeros(2574,18,5);
counter1 = 1;
counter2 = 1;

% Function to create folders to organize results
createFolders

for indexMain = 1:size(csvMain)
    
    % get file ID
    fileIndex = csvMain(indexMain,1);
    fileIndex = replace(fileIndex,'.mp4','');
    
    % get EMG for that ID
    fileRawEMG = strcat(fileIndex,'_EMG.txt');
    A = importdata(char(strcat('EMG\',fileRawEMG)));
    
    % get IMU for that ID
    fileRawIMU = strcat(fileIndex,'_IMU.txt');
    B = importdata(char(strcat('IMU\',fileRawIMU)));
    
    % get Annotation for that ID
    csvname = strcat(fileIndex,'.txt');
    annotationFile = importdata(char(strcat('Annotation\',csvname)));
    lastValue = annotationFile(size(annotationFile),2);
    
    % spoon file
    if rem(indexMain,2)==1
        
        scaleFactorEMG = 2;
        scaleFactorIMU = 1;
        
        % fork file
    else
        
        scaleFactorEMG = 3.3;
        scaleFactorIMU = 1.6;
        
    end
    
    % remove extra data from top
    choppedEMG = A((size(A)-floor(lastValue*scaleFactorEMG)+1):size(A),:);
    choppedIMU = B((size(B)-floor(lastValue*scaleFactorIMU)+1):size(B),:);
    
    %% Parse Annotation data for Eating and Non Eating
    
    for i = 1 : size(annotationFile)
        x1 = annotationFile(i,1);
        x2 = annotationFile(i,2);
        
        % Eating
        tempEat1 = choppedIMU(floor(x1*scaleFactorIMU):floor(x2*scaleFactorIMU),2:11);
        tempEat2 = choppedEMG(floor(x1*scaleFactorEMG):floor(x2*scaleFactorEMG),2:9);
        
        % feature extraction methods
        for fem= (1:5)
            resultEatMat(:,:,fem) = calcFeatMat(abs(fft(tempEat1)),abs(fft(tempEat2)),counter1,resultEatMat(:,:,fem),fem);
%             for rows = 1:size(resultEatMat(:,:,fem))
%                 for columns = 1:size(resultEatMat(:,:,fem),2)
%                     if isnan(resultEatMat(rows,columns,fem))
%                     disp(fileIndex);
%                     
%                     end
%                 end
%             end
        end
        
        counter1 = counter1+1;
        
        tempEat1 = transpose(tempEat1);
        tempEat2 = transpose(tempEat2);
        
        if(MaxColEating<((x2-x1)*scaleFactorEMG))
            MaxColEating=((x2-x1)*scaleFactorEMG);
        end
        
        % Copy IMU - Eating
        EatingAction(startPosEat:startPosEat + size(tempEat1) -1,1:size(tempEat1,2)) = tempEat1(:,:);
        startPosEat = startPosEat + size(tempEat1);
        
        % Copy EMG - Eating
        EatingAction(startPosEat:startPosEat + size(tempEat2) -1,1:size(tempEat2,2)) = tempEat2(:,:);
        startPosEat = startPosEat + size(tempEat2);
        
        if(i == size(annotationFile,1))
            break;
        end
        
        % Non Eating
        x3 = annotationFile(i+1,1);
        tempNonEat1 = choppedIMU(floor((x2+1)*scaleFactorIMU):floor((x3-1)*scaleFactorIMU),2:11);
        tempNonEat2 = choppedEMG(floor((x2+1)*scaleFactorEMG):floor((x3-1)*scaleFactorEMG),2:9);
        
        % feature extraction methods
        for nefem= (1:5)
            resultNonEatMat(:,:,nefem) = calcFeatMat(abs(fft(tempNonEat1)),abs(fft(tempNonEat2)),counter2,resultNonEatMat(:,:,nefem),nefem);
        end
        
        counter2 = counter2+1;
        tempNonEat1 = transpose(tempNonEat1);
        tempNonEat2 = transpose(tempNonEat2);
        
        % Copy IMU - Non Eating
        NonEatingAction(startPosNonEat:startPosNonEat + size(tempNonEat1) -1,1:size(tempNonEat1,2)) = tempNonEat1(:,:);
        startPosNonEat = startPosNonEat + size(tempNonEat1);
        
        % Copy EMG - Non Eating
        NonEatingAction(startPosNonEat:startPosNonEat + size(tempNonEat2) -1,1:size(tempNonEat2,2)) = tempNonEat2(:,:);
        startPosNonEat = startPosNonEat + size(tempNonEat2);
        
        if(MaxColNonEating<((x3-x2)*scaleFactorEMG))
            MaxColNonEating=((x3-x2)*scaleFactorEMG);
        end
    end
    
end

% Final data matrix
EatingAction = EatingAction(:,1:ceil(MaxColEating));
NonEatingAction = NonEatingAction(:,1:ceil(MaxColNonEating));

%% Write final data to csv (Task 1)
csvwrite('EatingAction.csv',EatingAction);
csvwrite('NonEatingAction.csv',NonEatingAction);

%% Calling Function To Write CSVs, Plot and Save Graphs (Task 2)
writeMatrixAndPlot(resultEatMat,resultNonEatMat);

%% Calling function to perform PCA, Plot and Save Graphs (Task 3)
PCAanalysis(resultEatMat(:,:,:),resultNonEatMat(:,:,:));

%% Task 2 - Write data matrix for five feature extraction methods used
function writeMatrixAndPlot(resultEatMat,resultNonEatMat)

names = {'variance','rms','kurtosis','skewness','std'};
features = {'_Accelerometer_X','_Accelerometer_Y','_Accelerometer_Z','_Gyroscope_X','_Gyroscope_Y','_Gyroscope_Z','_Orientation_X','_Orientation_Y','_Orientation_Z','_Orientation_W','EMG1','EMG2','EMG3','EMG4','EMG5','EMG6','EMG7','EMG8'};

% For Eating Actions
for it1 = 1:5
    EatCSVFldr = strcat(pwd,'\EatCSV\');
    fn = strcat(names(it1),'_Eat.csv');
    fn = strcat(EatCSVFldr,fn);
    csvwrite(char(fn),resultEatMat(:,:,it1));
end

% For Non Eating Actions
for it2 = 1:5
    NonEatCSVFldr = strcat(pwd,'\NonEatCSV\');
    fn = strcat(names(it2),'_NonEat.csv');
    fn = strcat(NonEatCSVFldr,fn);
    csvwrite(char(fn),resultNonEatMat(:,:,it1));
end

axis = 1:2574;

% Plot corresponding feature values - Eating vs Non Eating
for i=1:5
    mat1 = resultEatMat(:,:,i);
    mat2 = resultNonEatMat(:,:,i);
    for j=1:18
        x1 = mat1(1:2574,j);
        x2 = mat2(:,j);
        
        plot(axis,x1,'red')
        hold on
        plot(axis,x2,'blue');
        
        name = char(strcat(names(i),features(j)));
        GraphFldr = strcat(pwd,'\Graphs');
        fullFileName = fullfile(GraphFldr, name);
        
        saveas(gcf, fullFileName, 'jpg');
        cla reset;
    end
end
end

%% Function to calculate values based on feature Extraction Method
function resultMatr =  calcFeatMat(mIMU,mEMG,i,resultMatr, FEM)

% Variance
if(FEM == 1)
    extrFeature =  var(mIMU);
    extrFeature1 = var(mEMG);
end

% RMS
if(FEM == 2)
    extrFeature =  rms(mIMU);
    extrFeature1 = rms(mEMG);
end

% Kurtosis
if(FEM == 3)
    extrFeature =  kurtosis(mIMU);
    extrFeature1 = kurtosis(mEMG);
end

% Skewness
if(FEM == 4)
    extrFeature =  skewness(mIMU);
    extrFeature1 = skewness(mEMG);
end

% Standard Deviation
if(FEM == 5)
    extrFeature =  std(mIMU);
    extrFeature1 = std(mEMG);
end

r = [extrFeature extrFeature1];

resultMatr(i,:) = r;

end

%% Function To Create Folders
function createFolders

% Create the folder if it doesn't exist already
GraphFldr = strcat(pwd,'\Graphs');
if ~exist(GraphFldr, 'dir')
    mkdir(GraphFldr);
end

EatCSVFldr = strcat(pwd,'\EatCSV');
if ~exist(EatCSVFldr, 'dir')
    mkdir(EatCSVFldr);
end

NonEatCSVFldr = strcat(pwd,'\NonEatCSV');
if ~exist(NonEatCSVFldr, 'dir')
    mkdir(NonEatCSVFldr);
end

PCAGraphsFldr = strcat(pwd,'\PCAGraphs');
if ~exist(PCAGraphsFldr, 'dir')
    mkdir(PCAGraphsFldr);
end    
PCAEatCSVFldr = strcat(pwd,'\PCAEatCSV');
if ~exist(PCAEatCSVFldr, 'dir')
    mkdir(PCAEatCSVFldr);
end
PCANonEatCSVFldr = strcat(pwd,'\PCANonEatCSV');
if ~exist(PCANonEatCSVFldr, 'dir')
    mkdir(PCANonEatCSVFldr);
end
EigenVectorFldr = strcat(pwd,'\EigenVector');
if ~exist(EigenVectorFldr, 'dir')
    mkdir(EigenVectorFldr);
end
end

%% Task 3 - Perform PCA and plot graphs for comparison
function PCAanalysis(Eat,NonEat)
names = {'variance_','rms_','kurtosis_','skewness_','std_'};
for j = 1:5
% PCA
coeffEat = pca(Eat(:,:,j));
coeffNonEat = pca(NonEat(:,:,j));

axis1 = 1:18;

plot(axis1,diag(coeffEat),'red')
hold on
plot(axis1,diag(coeffNonEat),'blue')

EigenVectorFldr = strcat(pwd,'\EigenVector');
fullFileName = fullfile(EigenVectorFldr,strcat(names(j),'eigenVectors'));
saveas(gcf,char(fullFileName),'jpg')

cla reset;


% New feature matrix
NewEat = Eat(:,:,j)*coeffEat;
NewNonEat = NonEat(:,:,j)*coeffNonEat;

PCAEatCSVFldr = strcat(pwd,'\PCAEatCSV');
EatFilePath = char(strcat(names(j),'PCA.csv'));
fullFileName1 = fullfile(PCAEatCSVFldr, EatFilePath);
csvwrite(fullFileName1,NewEat);

PCANonEatCSVFldr = strcat(pwd,'\PCANonEatCSV');
NonEatFilePath = char(strcat(names(j),'PCA.csv'));
fullFileName2 = fullfile(PCANonEatCSVFldr, NonEatFilePath);
csvwrite(fullFileName2,NewNonEat);

% Plot
for i=1:18
    plot(1:2640,NewEat(:,i),'red')
    hold on
    plot(1:2574,NewNonEat(:,i),'blue')
    
    name = strcat('PrincipalComponent',int2str(i));
    name = char(strcat(names(j),name));
    PCAGraphsFldr = strcat(pwd,'\PCAGraphs');
    fullFileName = fullfile(PCAGraphsFldr, name);
    
    saveas(gcf, fullFileName, 'jpg');
    cla reset;
end
end
end