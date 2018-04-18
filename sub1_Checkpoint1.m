
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
load('Sub1_training_ecog.mat');
tdvFxn = @(x) mean(x);

xLen = 300000;
fs = 1000;
winLen = .1;
winDisp = .05;

%subject 1
sub1tdv = cell(1,62);
for i = 1:62
   sub1tdv{i} = MovingWinFeats(Sub1_Training_ecog{1,i,1}, fs, winLen, winDisp, tdvFxn);
end
%% Feature Extraction (Average Frequency-Domain Magnitude in 5 bands)
% Frequency bands are: 5-15Hz, 20-25Hz, 75-115Hz, 125-160Hz, 160-175Hz
% Total number of features in given time window is (num channels)*(5+1)
window = winLen*fs;
freq_arr = 0:5:500;
%subject 1
for i = 1:62
    [s,freq,t] = spectrogram(Sub1_Training_ecog{1,i,1},window,winDisp*fs,freq_arr,fs);
    sub1f5_15{i} = mean(s(2:4,:),1);
    sub1f20_25{i} = mean(s(5:6,:),1);
    sub1f75_115{i} = mean(s(16:24,:),1);
    sub1f125_160{i} = mean(s(26:32,:),1);
    sub1f160_175{i} = mean(s(32:36,:),1);
end

%% Decimation of dataglove
load('Sub1_Training_dg.mat');
% decimated glove data for subject one
% take out the last value to match our 5999
sub1DataGlove = cell(1,5);
for i = 1:5
    sub1DataGlove{i} = decimate(Sub1_Training_dg{i},50);
    sub1DataGlove{i}(end)= [];
end

%% Formation of the X matrix
% Referenced form HW7
% 62 channels ~ 40 neurons (HW7)
v = 62; % 62 channels
N = 3; % 3 time windows 
f = 6; % 6 features
sub1X = ones(5999,v*N*f+1);

% for m = 1:5999
%     disp(m);
%     X(m,:) = [1 reshape(sub1tdv{:}(m:(m+N-1)),1,62*N)];
% end
for j = 1:62
    %disp(j);
    for i = N:5999
        % error with sub1f20_25 input
    	sub1X(i,((j-1)*N*f+2):(j*N*f)+1) = [sub1tdv{j}(i-N+1:i) sub1f5_15{j}(i-N+1:i) sub1f20_25{j}(i-N+1:i) ...
            sub1f75_115{j}(i-N+1:i) sub1f125_160{j}(i-N+1:i) sub1f160_175{j}(i-N+1:i)]; %insert data into R
    end
end

sub1X(1:2,:) = [];
    
    %% Calculation - Shira
sub1fingerflexion = [sub1DataGlove{1} sub1DataGlove{2} sub1DataGlove{3} sub1DataGlove{4} sub1DataGlove{5}];
%sub1_weight = zeros(62*N+1,5);
sub1X = abs(sub1X);
arg1 = (sub1X'*sub1X);
arg2 = (sub1X'*sub1fingerflexion(N:end,:));
sub1_weight = mldivide(arg1,arg2);
sub1_trainpredict = sub1X*sub1_weight;

%% reconstruct R matrix for testing data, first get features then make R
%% Feature Extraction of TESTING (Average Time-Domain Voltage)
load('Sub1_Leaderboard_ecog.mat');
tdvFxn = @(x) mean(x);

xTestLen = 147500;
fs = 1000;
winLen = .1;
winDisp = .05;

%subject 1
sub1_testtdv = cell(1,62);
for i = 1:62
   sub1_testtdv{i} = MovingWinFeats(Sub1_Leaderboard_ecog{1,i,1}, fs, winLen, winDisp, tdvFxn);
end
%% Feature Extraction of TESTING (Average Frequency-Domain Magnitude in 5 bands)
% Frequency bands are: 5-15Hz, 20-25Hz, 75-115Hz, 125-160Hz, 160-175Hz
% Total number of features in given time window is (num channels)*(5+1)
window = winLen*fs;
freq_arr = 0:5:500;
%subject 1
for i = 1:62
    [s,freq,t] = spectrogram(Sub1_Leaderboard_ecog{1,i,1},window,winDisp*fs,freq_arr,fs);
    testsub1f5_15{i} = mean(s(2:4,:),1);
    testsub1f20_25{i} = mean(s(5:6,:),1);
    testsub1f75_115{i} = mean(s(16:24,:),1);
    testsub1f125_160{i} = mean(s(26:32,:),1);
    testsub1f160_175{i} = mean(s(32:36,:),1);
end

%% Formation of the X matrix but now its for testing set
% Referenced form HW7
% 62 channels ~ 40 neurons (HW7)
v = 62; % 62 channels
N = 3; % 3 time windows 
f = 6; % 6 features
xLen = 147500;
fs = 1000;
winLen = 100 * 1e-3;
winDisp = 50 * 1e-3;
NumWins = @(xLen, fs, winLen, winDisp) length(0:winDisp*fs:xLen)-(winLen/winDisp);

testsub1X = ones(NumWins(xLen,fs,winLen,winDisp),v*N*f+1);

for j = 1:62
    %disp(j);
    for i = N:NumWins(xLen,fs,winLen,winDisp)
       
    	testsub1X(i,((j-1)*N*f+2):(j*N*f)+1) = [sub1_testtdv{j}(i-N+1:i) testsub1f5_15{j}(i-N+1:i) testsub1f20_25{j}(i-N+1:i) ...
            testsub1f75_115{j}(i-N+1:i) testsub1f125_160{j}(i-N+1:i) testsub1f160_175{j}(i-N+1:i)]; %insert data into R
    end
end

testsub1X(1:2,:) = [];

%% make the prediction using the testing R matrix
testsub1X = abs(testsub1X);
sub1_testpredict = testsub1X*sub1_weight;

%% spline stuff
% will zero pad at the end
% [1:lastSample].*50 to reconstruct as much as we can then pad to 150k pt
% sub1_predict is our prediction on our testing data
% which will be 50th sample to the 2947*50th sample
sub1Spline = spline(50.*(1:length(sub1_testpredict)),sub1_testpredict',(50:50*length(sub1_testpredict)));
% remember to un-transpose sub1_testpredict at the end

