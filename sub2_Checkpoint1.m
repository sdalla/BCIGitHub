
%% Feature Extraction Code (taken from HW3)
% gets the number of windows given the length and displacement
% winLen and winDisp are in time(s), fs is sampling freq, xLen is length in
% samples
xLen = 300000;
fs = 1000;
winLen = 100 * 1e-3;
winDisp = 50 * 1e-3;
NumWins = @(xLen, fs, winLen, winDisp) length(0:winDisp*fs:xLen)-(winLen/winDisp);

% Line length fxn
LLFn = @(x) sum(abs(diff(x)));
% Area Fxn
areaFxn = @(x) sum(abs(x));
% Energy Fxn
energyFxn =@(x) sum(x.^2);
% Zero crossings around mean
zxFxn = @(x) sum((x(1:end-1)-mean(x)).*(x(2:end)-mean(x))<=0);
%% Feature Extraction (Average Time-Domain Voltage)
load('Sub2_training_ecog.mat');
tdvFxn = @(x) mean(x);

xLen = 300000;
fs = 1000;
winLen = .1;
winDisp = .05;

%subject 1
sub2tdv = cell(1,48);
for i = 1:48
   sub2tdv{i} = MovingWinFeats(Sub2_Training_ecog{1,i,1}, fs, winLen, winDisp, tdvFxn);
end
%% Feature Extraction (Average Frequency-Domain Magnitude in 5 bands)
% Frequency bands are: 5-15Hz, 20-25Hz, 75-115Hz, 125-160Hz, 160-175Hz
% Total number of features in given time window is (num channels)*(5+1)
window = winLen*fs;
freq_arr = 0:5:500;
%subject 1
for i = 1:48
    [s,freq,t] = spectrogram(Sub2_Training_ecog{1,i,1},window,winDisp*fs,freq_arr,fs);
    sub2f5_15{i} = mean(s(2:4,:),1);
    sub2f20_25{i} = mean(s(5:6,:),1);
    sub2f75_115{i} = mean(s(16:24,:),1);
    sub2f125_160{i} = mean(s(26:32,:),1);
    sub2f160_175{i} = mean(s(32:36,:),1);
end

%% Decimation of dataglove
load('Sub2_Training_dg.mat');
% decimated glove data for subject one
% take out the last value to match our 5999
sub2DataGlove = cell(1,5);
for i = 1:5
    sub2DataGlove{i} = decimate(Sub2_Training_dg{i},50);
    sub2DataGlove{i}(end)= [];
end

%% Formation of the X matrix
% Referenced form HW7
% 48 channels ~ 40 neurons (HW7)
v = 48; % 48 channels
N = 3; % 3 time windows 
f = 6; % 6 features
sub2X = ones(5999,v*N*f+1);

% for m = 1:5999
%     disp(m);
%     X(m,:) = [1 reshape(sub2tdv{:}(m:(m+N-1)),1,48*N)];
% end
for j = 1:48
    %disp(j);
    for i = N:5999
        % error with sub2f20_25 input
    	sub2X(i,((j-1)*N*f+2):(j*N*f)+1) = [sub2tdv{j}(i-N+1:i) sub2f5_15{j}(i-N+1:i) sub2f20_25{j}(i-N+1:i) ...
            sub2f75_115{j}(i-N+1:i) sub2f125_160{j}(i-N+1:i) sub2f160_175{j}(i-N+1:i)]; %insert data into R
    end
end

sub2X(1:2,:) = [];
    
    %% Calculation - Shira
sub2fingerflexion = [sub2DataGlove{1} sub2DataGlove{2} sub2DataGlove{3} sub2DataGlove{4} sub2DataGlove{5}];
%sub2_weight = zeros(48*N+1,5);
sub2X = abs(sub2X);
arg1 = (sub2X'*sub2X);
arg2 = (sub2X'*sub2fingerflexion(N:end,:));
sub2_weight = mldivide(arg1,arg2);
sub2_trainpredict = sub2X*sub2_weight;

%% reconstruct R matrix for testing data, first get features then make R
%% Feature Extraction of TESTING (Average Time-Domain Voltage)
load('Sub2_Leaderboard_ecog.mat');
tdvFxn = @(x) mean(x);

xTestLen = 147500;
fs = 1000;
winLen = .1;
winDisp = .05;

%subject 1
sub2_testtdv = cell(1,48);
for i = 1:48
   sub2_testtdv{i} = MovingWinFeats(Sub2_Leaderboard_ecog{1,i,1}, fs, winLen, winDisp, tdvFxn);
end
%% Feature Extraction of TESTING (Average Frequency-Domain Magnitude in 5 bands)
% Frequency bands are: 5-15Hz, 20-25Hz, 75-115Hz, 125-160Hz, 160-175Hz
% Total number of features in given time window is (num channels)*(5+1)
window = winLen*fs;
freq_arr = 0:5:500;
%subject 1
for i = 1:48
    [s,freq,t] = spectrogram(Sub2_Leaderboard_ecog{1,i,1},window,winDisp*fs,freq_arr,fs);
    testsub2f5_15{i} = mean(s(2:4,:),1);
    testsub2f20_25{i} = mean(s(5:6,:),1);
    testsub2f75_115{i} = mean(s(16:24,:),1);
    testsub2f125_160{i} = mean(s(26:32,:),1);
    testsub2f160_175{i} = mean(s(32:36,:),1);
end

%% Formation of the X matrix but now its for testing set
% Referenced form HW7
% 48 channels ~ 40 neurons (HW7)
v = 48; % 48 channels
N = 3; % 3 time windows 
f = 6; % 6 features
xLen = 147500;
fs = 1000;
winLen = 100 * 1e-3;
winDisp = 50 * 1e-3;
NumWins = @(xLen, fs, winLen, winDisp) length(0:winDisp*fs:xLen)-(winLen/winDisp);

testsub2X = ones(NumWins(xLen,fs,winLen,winDisp),v*N*f+1);

for j = 1:48
    %disp(j);
    for i = N:NumWins(xLen,fs,winLen,winDisp)
       
    	testsub2X(i,((j-1)*N*f+2):(j*N*f)+1) = [sub2_testtdv{j}(i-N+1:i) testsub2f5_15{j}(i-N+1:i) testsub2f20_25{j}(i-N+1:i) ...
            testsub2f75_115{j}(i-N+1:i) testsub2f125_160{j}(i-N+1:i) testsub2f160_175{j}(i-N+1:i)]; %insert data into R
    end
end

testsub2X(1:2,:) = [];

%% make the prediction using the testing R matrix
testsub2X = abs(testsub2X);
sub2_testpredict = testsub2X*sub2_weight;

%% spline stuff
% will zero pad at the end
% [1:lastSample].*50 to reconstruct as much as we can then pad to 150k pt
% sub2_predict is our prediction on our testing data
% which will be 50th sample to the 2947*50th sample
sub2Spline = spline(50.*(1:length(sub2_testpredict)),sub2_testpredict',(50:50*length(sub2_testpredict)));
% remember to un-transpose sub2_testpredict at the end
sub2Pad = padarray(sub2Spline, [0 99]);
sub2Pad(:,end+1) = 0;
sub2Final = sub2Pad';
