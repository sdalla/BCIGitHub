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
load B.mat
load intercept1.mat
load Sub1_Leaderboard_ecog
load Sub2_Leaderboard_ecog
load Sub3_Leaderboard_ecog
test_ecog = {Sub1_Leaderboard_ecog Sub2_Leaderboard_ecog Sub3_Leaderboard_ecog};

%% Pre-processing ecog signal
% Sub 1 Channel Elimination
ch_remove1 = [55 21 44 52 18 27 40 49]; %remove channels found in other mat file
channel1 = 1:62;
channel1(ch_remove1) = [];
ind = 1;
for j = channel1
    Sub1_ecog_ch{ind} = cell2mat(test_ecog{1}(:,j));
    ind = ind + 1;
end
v1 = size(Sub1_ecog_ch,2);

% Sub 2 Channel Elimination
ch_remove2 = [23 37 38];
channel2 = 1:48;
channel2(ch_remove2) = [];
ind = 1;
for j = channel2
    Sub2_ecog_ch{ind} = cell2mat(test_ecog{2}(:,j));
    ind = ind + 1;
end
v2 = size(Sub2_ecog_ch,2);

% Sub 3 Channel Elimination
ch_remove3 = [23 38 58];
channel3 = 1:64;
channel3(ch_remove3) = [];
ind = 1;
for j = channel3
    Sub3_ecog_ch{ind} = cell2mat(test_ecog{3}(:,j));
    ind = ind + 1;
end
v3 = size(Sub3_ecog_ch,2);

%% Feature extraction
% Filter data
load('Filter1.mat');
%Filter1 = temp.Filter1;
load('Filter2.mat');
%Filter2 = temp.Filter2;
load('Filter3.mat');
%Filter3 = temp.Filter3;

% filter subject 1
for i = 1:v1
    sub1_1_60filt{i} = filtfilt(Filter1,1.0,Sub1_ecog_ch{i});
    sub1_60_100filt{i} = filtfilt(Filter2,1.0,Sub1_ecog_ch{i});
    sub1_100_200filt{i} = filtfilt(Filter3,1.0,Sub1_ecog_ch{i});
end

% filter subject 2
for k = 1:v2
    sub2_1_60filt{k} = filtfilt(Filter1,1.0,Sub2_ecog_ch{k});
    sub2_60_100filt{k} = filtfilt(Filter2,1.0,Sub2_ecog_ch{k});
    sub2_100_200filt{k} = filtfilt(Filter3,1.0,Sub2_ecog_ch{k});
end

% filter subject 3
for i = 1:v3
    sub3_1_60filt{i} = filtfilt(Filter1,1.0,Sub3_ecog_ch{i});
    sub3_60_100filt{i} = filtfilt(Filter2,1.0,Sub3_ecog_ch{i});
    sub3_100_200filt{i} = filtfilt(Filter3,1.0,Sub3_ecog_ch{i});
end

%% Amplitude Modulation for all Subjects
f = 3; % 3 features
N = 4;
xLen = size(Sub3_ecog_ch{1,1},1);
fs = 1000;
winLen = 100 * 1e-3;
winDisp = 50 * 1e-3;
NumWins = @(xLen, fs, winLen, winDisp) length(0:winDisp*fs:xLen)-(winLen/winDisp);
ampFxn = @(x) sum(x.^2);

for i = 1:v1
    sub1_1_60amp{i} = MovingWinFeats(sub1_1_60filt{i}, fs, winLen, winDisp, ampFxn);
    sub1_60_100amp{i} = MovingWinFeats(sub1_60_100filt{i}, fs, winLen, winDisp, ampFxn);
    sub1_100_200amp{i} = MovingWinFeats(sub1_100_200filt{i}, fs, winLen, winDisp, ampFxn);
end

for i = 1:v2
    sub2_1_60amp{i} = MovingWinFeats(sub2_1_60filt{i}, fs, winLen, winDisp, ampFxn);
    sub2_60_100amp{i} = MovingWinFeats(sub2_60_100filt{i}, fs, winLen, winDisp, ampFxn);
    sub2_100_200amp{i} = MovingWinFeats(sub2_100_200filt{i}, fs, winLen, winDisp, ampFxn);
end

for i = 1:v3
    sub3_1_60amp{i} = MovingWinFeats(sub3_1_60filt{i}, fs, winLen, winDisp, ampFxn);
    sub3_60_100amp{i} = MovingWinFeats(sub3_60_100filt{i}, fs, winLen, winDisp, ampFxn);
    sub3_100_200amp{i} = MovingWinFeats(sub3_100_200filt{i}, fs, winLen, winDisp, ampFxn);
end

% Construct X matrices
% sub 1 X matrix
sub1X = ones(NumWins(xLen,fs,winLen,winDisp),v1*N*f+1);

for j = 1:v1
    for i = N:NumWins(xLen,fs,winLen,winDisp)
        sub1X(i,((j-1)*N*f+2):(j*N*f)+1) = [sub1_1_60amp{j}(i-N+1:i) sub1_60_100amp{j}(i-N+1:i) ...
            sub1_100_200amp{j}(i-N+1:i)]; %insert data into R
    end
end

sub1X(1:N-1,:) = [];

% sub 2 X matrix
sub2X = ones(NumWins(xLen,fs,winLen,winDisp),v2*N*f+1);

for j = 1:v2
    for i = N:NumWins(xLen,fs,winLen,winDisp)
        sub2X(i,((j-1)*N*f+2):(j*N*f)+1) = [sub2_1_60amp{j}(i-N+1:i) sub2_60_100amp{j}(i-N+1:i) ...
            sub2_100_200amp{j}(i-N+1:i)]; %insert data into R
    end
end

sub2X(1:N-1,:) = [];

% sub3
sub3X = ones(NumWins(xLen,fs,winLen,winDisp),v3*N*f+1);

for j = 1:v3
    %disp(j);
    for i = N:NumWins(xLen,fs,winLen,winDisp)
        sub3X(i,((j-1)*N*f+2):(j*N*f)+1) = [sub3_1_60amp{j}(i-N+1:i) sub3_60_100amp{j}(i-N+1:i) ...
            sub3_100_200amp{j}(i-N+1:i)]; %insert data into R
    end
end

sub3X(1:N-1,:) = [];

X = {sub1X sub2X sub3X};
%% Linear prediction & post-processing
% Predict using linear predictor for each subject
%create cell array with one element for each subject
predicted_dg = cell(3,1);

%for each subject
for subj = 1:3 
    
    testset = X{subj}; 
    
    %initialize the predicted dataglove matrix
    yhat = zeros(size(test_ecog{1},1),5);
    
    %for each finger
    for i = 1:5 
        if i == 4
            yhat(:,i) = 0;
            continue
        end
        % predict dg based on ECOG for each finger
        predy = testset*B{i,subj} + repmat(Intercept{i,subj},size((testset*B{i,subj}),1),1);
        predy = predy(:,1);
        
        % spline the data
        subSpline = spline(50.*(1:length(predy)),predy',(50:50*length(predy)));
        padSpline = [zeros(1,200) subSpline zeros(1,49)];
        
        % filter predicted finger positions
        yhat(:,i) = medfilt1(padSpline,1000);
        
    end
    predicted_dg{subj} = yhat;
end

