%for this data set, the band-specific ECoG signals were generated using 
%equiripple finite impulse response (FIR) filters by setting their band-pass
%specifications as: sub-bands (1-60 Hz), gamma band (60-100 Hz), and fast 
%gamma band (100-200 Hz)
%testing on sub3 only, for now
% load data
xLen = 300000;
fs = 1000;
winLen = 100 * 1e-3;

winDisp = 50 * 1e-3;
NumWins = @(xLen, fs, winLen, winDisp) length(0:winDisp*fs:xLen)-(winLen/winDisp);
%ecog data
load('Sub3_training_ecog.mat');
%original dataglove
load('sub3_Training_dg.mat');
%decimated dataglove
load('sub3DataGlove.mat');
%leaderboard ecog
load('Sub3_Leaderboard_ecog.mat');

ch_remove = [23 38 58]; %remove channels found in other mat file
v = 64 - length(ch_remove);

channel = 1:64;
%channel(ch_remove) = [];
ind = 1;
for j = channel
    Sub3_Training_ecog_ch{ind} = Sub3_Training_ecog{j};
    Sub3_Test_ecog_ch{ind} = Sub3_Leaderboard_ecog{j};
    ind = ind + 1;
end
tdvFxn = @(x) mean(x);

xLen = 300000;
fs = 1000;
winLen = .1;
winDisp = .05;

%subject 1
sub3tdv = cell(1,v);
for i = 1:v
   sub3tdv{i} = MovingWinFeats(Sub3_Training_ecog_ch{1,i}, fs, winLen, winDisp, tdvFxn);
end

sub3tdvTest = cell(1,v);
for i = 1:v
   sub3tdvTest{i} = MovingWinFeats(Sub3_Test_ecog_ch{1,i}, fs, winLen, winDisp, tdvFxn);
end
%% Feature Extraction (Average Frequency-Domain Magnitude in 5 bands)
% Frequency bands are: 5-15Hz, 20-25Hz, 75-115Hz, 125-160Hz, 160-175Hz
% Total number of features in given time window is (num channels)*(5+1)
window = winLen*fs;
freq_arr = 0:1:500; 
%subject 1
for i = 1:v
    [s,freq,t] = spectrogram(Sub3_Training_ecog_ch{1,i},window,winDisp*fs,freq_arr,fs);
    sub3f5_15{i} = mean(abs(s(6:16,:)),1);
    sub3f20_25{i} = mean(abs(s(21:26,:)),1);
    sub3f75_115{i} = mean(abs(s(76:116,:)),1);
    sub3f125_160{i} = mean(abs(s(126:161,:)),1);
    sub3f160_175{i} = mean(abs(s(161:176,:)),1);
end
%Test spec
for i = 1:v
    [s,freq,t] = spectrogram(Sub3_Test_ecog_ch{1,i},window,winDisp*fs,freq_arr,fs);
    testsub3f5_15{i} = mean(abs(s(6:16,:)),1);
    testsub3f20_25{i} = mean(abs(s(21:26,:)),1);
    testsub3f75_115{i} = mean(abs(s(76:116,:)),1);
    testsub3f125_160{i} = mean(abs(s(126:161,:)),1);
    testsub3f160_175{i} = mean(abs(s(161:176,:)),1);
end

% Constructing a new X (R) matrix

v = 64-3; % 62 channels
N = 3; % time windows 
f = 6; % 6 features
sub3X = ones(5999,v*N*f+1);
sub3XTest = ones(5999,v*N*f+1);


for j = 1:v
    %disp(j);
    for i = N:5999
        % error with sub3f20_25 input
    	sub3X(i,((j-1)*N*f+2):(j*N*f)+1) = [sub3tdv{j}(i-N+1:i) sub3f5_15{j}(i-N+1:i) sub3f20_25{j}(i-N+1:i) ...
            sub3f75_115{j}(i-N+1:i) sub3f125_160{j}(i-N+1:i) sub3f160_175{j}(i-N+1:i)]; %insert data into R
    end
end



sub3X(1:N-1,:) = [];



% Test R
v = 64-3; % 62 channels
N = 3; % time windows 
f = 6; % 6 features
xLen = 147500;
fs = 1000;
winLen = 100 * 1e-3;
winDisp = 50 * 1e-3;
NumWins = @(xLen, fs, winLen, winDisp) length(0:winDisp*fs:xLen)-(winLen/winDisp);

sub3XTest = ones(NumWins(xLen,fs,winLen,winDisp),v*N*f+1);

for j = 1:v
    %disp(j);
    for i = N:NumWins(xLen,fs,winLen,winDisp)
       
    	sub3XTest(i,((j-1)*N*f+2):(j*N*f)+1) = [sub3tdvTest{j}(i-N+1:i) testsub3f5_15{j}(i-N+1:i) testsub3f20_25{j}(i-N+1:i) ...
            testsub3f75_115{j}(i-N+1:i) testsub3f125_160{j}(i-N+1:i) testsub3f160_175{j}(i-N+1:i)]; %insert data into R
    end
end

sub3XTest(1:2,:) = [];
% Calculation 
sub3fingerflexion = [sub3DataGlove{1} sub3DataGlove{2} sub3DataGlove{3} sub3DataGlove{4} sub3DataGlove{5}];


[B1, FitInfo] = lasso(sub3X,sub3fingerflexion(N:end,1));
lassTestPred1 = sub3XTest*B1 + repmat(FitInfo.Intercept,size((sub3XTest*B1),1),1);

disp('lasso 1 done')
[B2, FitInfo] = lasso(sub3X,sub3fingerflexion(N:end,2));
lassTestPred2 = sub3XTest*B2 + repmat(FitInfo.Intercept,size((sub3XTest*B2),1),1);

disp('lasso 2 done')
[B3, FitInfo] = lasso(sub3X,sub3fingerflexion(N:end,3));
lassTestPred3 = sub3XTest*B3 + repmat(FitInfo.Intercept,size((sub3XTest*B3),1),1);



[B5, FitInfo] = lasso(sub3X,sub3fingerflexion(N:end,5));
lassTestPred5 = sub3XTest*B5 + repmat(FitInfo.Intercept,size((sub3XTest*B5),1),1);


lassTestPred1 = lassTestPred1(:,1);
lassTestPred2 = lassTestPred2(:,1);
lassTestPred3 = lassTestPred3(:,1);
lassTestPred5 = lassTestPred5(:,1);

% spline
sub3Spline1 = spline(50.*(1:length(lassTestPred1)),lassTestPred1',(50:50*length(lassTestPred1)));
sub3Pad1 = [zeros(1,150) sub3Spline1 zeros(1,49)];
sub3Final1 = sub3Pad1';

sub3Spline2 = spline(50.*(1:length(lassTestPred2)),lassTestPred2',(50:50*length(lassTestPred2)));
sub3Pad2 = [zeros(1,150) sub3Spline2 zeros(1,49)];
sub3Final2 = sub3Pad2';

sub3Spline3 = spline(50.*(1:length(lassTestPred3)),lassTestPred3',(50:50*length(lassTestPred3)));
sub3Pad3 = [zeros(1,150) sub3Spline3 zeros(1,49)];
sub3Final3 = sub3Pad3';

sub3Spline5 = spline(50.*(1:length(lassTestPred5)),lassTestPred5',(50:50*length(lassTestPred5)));
sub3Pad5 = [zeros(1,150) sub3Spline5 zeros(1,49)];
sub3Final5 = sub3Pad5';

sub3chp2 = [sub3Final1 sub3Final2 sub3Final3 zeros(147500,1) sub3Final5];
save sub3checkpoint2ft6 sub3chp2