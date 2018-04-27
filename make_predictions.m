function [predicted_dg] = make_predictions(test_ecog)

%
% Inputs: test_ecog - 3 x 1 cell array containing ECoG for each subject, where test_ecog{i} 
% to the ECoG for subject i. Each cell element contains a N x M testing ECoG,
% where N is the number of samples and M is the number of EEG channels.
% Outputs: predicted_dg - 3 x 1 cell array, where predicted_dg{i} contains the 
% data_glove prediction for subject i, which is an N x 5 matrix (for
% fingers 1:5)
% Run time: The script has to run less than 1 hour.
%
% The following is a sample script.

% Load Model
% Imagine this mat file has the following variables:
% winDisp, filtTPs, trainFeats (cell array), 


% mat file will contain:
% (1) a 5x3 (finger x subject) cell array containing the B lasso output for 
% each subject and finger - called B
% (2) a 5x3 (finger x subject) cell array each containing the lasso output
% for incercept output for each subject and finger - called intercept
load TWTIFN.mat 

%% Pre-processing ecog signal
% Sub 1 Channel Elimination
% Sub 2 Channel Elimination
% Sub 3 Channel Elimination
ch_remove = [23 38 58]; %remove channels found in other mat file
channel = 1:64;
channel(ch_remove) = [];
ind = 1;
for j = channel
    Sub3_Test_ecog_ch{ind} = test_ecog{j};
    ind = ind + 1;
end
v3 = size(Sub3_Training_ecog_ch,2);

%% Feature extraction
% Sub 1 feature extraction
% Sub 2 feature extraction
% Sub 3 feature extraction


%% Linear prediction & post-processing
% Predict using linear predictor for each subject
%create cell array with one element for each subject
predicted_dg = cell(3,1);

%for each subject
for subj = 1:3 
    
    %get the testing ecog
    testset = test_ecog{subj}; 
    
    %initialize the predicted dataglove matrix
    yhat = zeros(size(testset,1),5);
    
    %for each finger
    for i = 1:5 
        if i == 4
            yhat(:,i) = 0;
        end
        % predict dg based on ECOG for each finger
        predy = testset(:,1)*B{i,subj} + repmat(intercept{i,subj},size((testset(:,1)*B{i,subj}),1),1);
        
        % spline the data
        subSpline = spline(50.*(1:length(predy)),predy',(50:50*length(predy)));
        padSpline = [zeros(1,200) subSpline zeros(1,49)];
        
        % filter predicted finger positions
        yhat(:,i) = medfilt1(padSpline,1000);
        
    end
    predicted_dg{subj} = yhat;
end

