%for this data set, the band-specific ECoG signals were generated using 
%equiripple finite impulse response (FIR) filters by setting their band-pass
%specifications as: sub-bands (1-60 Hz), gamma band (60-100 Hz), and fast 
%gamma band (100-200 Hz)
%testing on sub2 only, for now
% load data
xLen = 300000;
fs = 1000;
winLen = 100 * 1e-3;
winDisp = 50 * 1e-3;
NumWins = @(xLen, fs, winLen, winDisp) length(0:winDisp*fs:xLen)-(winLen/winDisp);
%ecog data
load('Sub2_training_ecog.mat');
%original dataglove
load('sub2_Training_dg.mat');
%decimated dataglove
load('sub2DataGlove.mat');
%leaderboard ecog
load('Sub2_Leaderboard_ecog.mat');

ch_remove = [23 37 38]; %remove channels found in other mat file
v = 48 - length(ch_remove);

channel = 1:48;
%channel(ch_remove) = [];
ind = 1;
for j = channel
    Sub2_Training_ecog_ch{ind} = Sub2_Training_ecog{j};
    Sub2_Test_ecog_ch{ind} = Sub2_Leaderboard_ecog{j};
    ind = ind + 1;
end
tdvFxn = @(x) mean(x);

xLen = 300000;
fs = 1000;
winLen = .1;
winDisp = .05;

%subject 1
sub2tdv = cell(1,v);
for i = 1:v
   sub2tdv{i} = MovingWinFeats(Sub2_Training_ecog_ch{1,i}, fs, winLen, winDisp, tdvFxn);
end

sub2tdvTest = cell(1,v);
for i = 1:v
   sub2tdvTest{i} = MovingWinFeats(Sub2_Test_ecog_ch{1,i}, fs, winLen, winDisp, tdvFxn);
end
%% Feature Extraction (Average Frequency-Domain Magnitude in 5 bands)
% Frequency bands are: 5-15Hz, 20-25Hz, 75-115Hz, 125-160Hz, 160-175Hz
% Total number of features in given time window is (num channels)*(5+1)
window = winLen*fs;
freq_arr = 0:1:500; 
%subject 1
for i = 1:v
    [s,freq,t] = spectrogram(Sub2_Training_ecog_ch{1,i},window,winDisp*fs,freq_arr,fs);
    sub2f5_15{i} = mean(abs(s(6:16,:)),1);
    sub2f20_25{i} = mean(abs(s(21:26,:)),1);
    sub2f75_115{i} = mean(abs(s(76:116,:)),1);
    sub2f125_160{i} = mean(abs(s(126:161,:)),1);
    sub2f160_175{i} = mean(abs(s(161:176,:)),1);
end
%Test spec
for i = 1:v
    [s,freq,t] = spectrogram(Sub2_Test_ecog_ch{1,i},window,winDisp*fs,freq_arr,fs);
    testsub2f5_15{i} = mean(abs(s(6:16,:)),1);
    testsub2f20_25{i} = mean(abs(s(21:26,:)),1);
    testsub2f75_115{i} = mean(abs(s(76:116,:)),1);
    testsub2f125_160{i} = mean(abs(s(126:161,:)),1);
    testsub2f160_175{i} = mean(abs(s(161:176,:)),1);
end

% Constructing a new X (R) matrix

v = 48-3; % 62 channels
N = 3; % time windows 
f = 6; % 6 features
sub2X = ones(5999,v*N*f+1);
sub2XTest = ones(5999,v*N*f+1);


for j = 1:v
    %disp(j);
    for i = N:5999
        % error with sub2f20_25 input
    	sub2X(i,((j-1)*N*f+2):(j*N*f)+1) = [sub2tdv{j}(i-N+1:i) sub2f5_15{j}(i-N+1:i) sub2f20_25{j}(i-N+1:i) ...
            sub2f75_115{j}(i-N+1:i) sub2f125_160{j}(i-N+1:i) sub2f160_175{j}(i-N+1:i)]; %insert data into R
    end
end



sub2X(1:N-1,:) = [];



% Test R
v = 48-3; % 62 channels
N = 3; % time windows 
f = 6; % 6 features
xLen = 147500;
fs = 1000;
winLen = 100 * 1e-3;
winDisp = 50 * 1e-3;
NumWins = @(xLen, fs, winLen, winDisp) length(0:winDisp*fs:xLen)-(winLen/winDisp);

sub2XTest = ones(NumWins(xLen,fs,winLen,winDisp),v*N*f+1);

for j = 1:v
    %disp(j);
    for i = N:NumWins(xLen,fs,winLen,winDisp)
       
    	sub2XTest(i,((j-1)*N*f+2):(j*N*f)+1) = [sub2tdvTest{j}(i-N+1:i) testsub2f5_15{j}(i-N+1:i) testsub2f20_25{j}(i-N+1:i) ...
            testsub2f75_115{j}(i-N+1:i) testsub2f125_160{j}(i-N+1:i) testsub2f160_175{j}(i-N+1:i)]; %insert data into R
    end
end

sub2XTest(1:2,:) = [];
% Calculation 
sub2fingerflexion = [sub2DataGlove{1} sub2DataGlove{2} sub2DataGlove{3} sub2DataGlove{4} sub2DataGlove{5}];


[B1, FitInfo] = lasso(sub2X,sub2fingerflexion(N:end,1));
lassTestPred1 = sub2XTest*B1 + repmat(FitInfo.Intercept,size((sub2XTest*B1),1),1);

disp('lasso 1 done')
[B2, FitInfo] = lasso(sub2X,sub2fingerflexion(N:end,2));
lassTestPred2 = sub2XTest*B2 + repmat(FitInfo.Intercept,size((sub2XTest*B2),1),1);

disp('lasso 2 done')
[B3, FitInfo] = lasso(sub2X,sub2fingerflexion(N:end,3));
lassTestPred3 = sub2XTest*B3 + repmat(FitInfo.Intercept,size((sub2XTest*B3),1),1);



[B5, FitInfo] = lasso(sub2X,sub2fingerflexion(N:end,5));
lassTestPred5 = sub2XTest*B5 + repmat(FitInfo.Intercept,size((sub2XTest*B5),1),1);


lassTestPred1 = lassTestPred1(:,1);
lassTestPred2 = lassTestPred2(:,1);
lassTestPred3 = lassTestPred3(:,1);
lassTestPred5 = lassTestPred5(:,1);

% spline
sub2Spline1 = spline(50.*(1:length(lassTestPred1)),lassTestPred1',(50:50*length(lassTestPred1)));
sub2Pad1 = [zeros(1,150) sub2Spline1 zeros(1,49)];
sub2Final1 = sub2Pad1';

sub2Spline2 = spline(50.*(1:length(lassTestPred2)),lassTestPred2',(50:50*length(lassTestPred2)));
sub2Pad2 = [zeros(1,150) sub2Spline2 zeros(1,49)];
sub2Final2 = sub2Pad2';

sub2Spline3 = spline(50.*(1:length(lassTestPred3)),lassTestPred3',(50:50*length(lassTestPred3)));
sub2Pad3 = [zeros(1,150) sub2Spline3 zeros(1,49)];
sub2Final3 = sub2Pad3';

sub2Spline5 = spline(50.*(1:length(lassTestPred5)),lassTestPred5',(50:50*length(lassTestPred5)));
sub2Pad5 = [zeros(1,150) sub2Spline5 zeros(1,49)];
sub2Final5 = sub2Pad5';

sub2chp2 = [sub2Final1 sub2Final2 sub2Final3 zeros(147500,1) sub2Final5];
save sub2checkpoint2ft6 sub2chp2