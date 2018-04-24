%for this data set, the band-specific ECoG signals were generated using 
%equiripple finite impulse response (FIR) filters by setting their band-pass
%specifications as: sub-bands (1-60 Hz), gamma band (60-100 Hz), and fast 
%gamma band (100-200 Hz)
%testing on sub1 only, for now
% load data
xLen = 300000;
fs = 1000;
winLen = 100 * 1e-3;
winDisp = 50 * 1e-3;
NumWins = @(xLen, fs, winLen, winDisp) length(0:winDisp*fs:xLen)-(winLen/winDisp);
%ecog data
load('Sub1_training_ecog.mat');
%original dataglove
load('sub1_Training_dg.mat');
%decimated dataglove
load('sub1DataGlove.mat');
%leaderboard ecog
load('Sub1_Leaderboard_ecog.mat');

ch_remove = [55 21 44 52 18 27 40 49]; %remove channels found in other mat file
v = 62 - length(ch_remove);

channel = 1:62;
%channel(ch_remove) = [];
ind = 1;
for j = channel
    Sub1_Training_ecog_ch{ind} = Sub1_Training_ecog{j};
    Sub1_Test_ecog_ch{ind} = Sub1_Leaderboard_ecog{j};
    ind = ind + 1;
end
tdvFxn = @(x) mean(x);

xLen = 300000;
fs = 1000;
winLen = .1;
winDisp = .05;

%subject 1
sub1tdv = cell(1,v);
for i = 1:v
   sub1tdv{i} = MovingWinFeats(Sub1_Training_ecog_ch{1,i}, fs, winLen, winDisp, tdvFxn);
end

sub1tdvTest = cell(1,v);
for i = 1:v
   sub1tdvTest{i} = MovingWinFeats(Sub1_Test_ecog_ch{1,i}, fs, winLen, winDisp, tdvFxn);
end
%% Feature Extraction (Average Frequency-Domain Magnitude in 5 bands)
% Frequency bands are: 5-15Hz, 20-25Hz, 75-115Hz, 125-160Hz, 160-175Hz
% Total number of features in given time window is (num channels)*(5+1)
window = winLen*fs;
freq_arr = 0:1:500; 
%subject 1
for i = 1:v
    [s,freq,t] = spectrogram(Sub1_Training_ecog_ch{1,i},window,winDisp*fs,freq_arr,fs);
    sub1f5_15{i} = mean(abs(s(6:16,:)),1);
    sub1f20_25{i} = mean(abs(s(21:26,:)),1);
    sub1f75_115{i} = mean(abs(s(76:116,:)),1);
    sub1f125_160{i} = mean(abs(s(126:161,:)),1);
    sub1f160_175{i} = mean(abs(s(161:176,:)),1);
end
%Test spec
for i = 1:v
    [s,freq,t] = spectrogram(Sub1_Test_ecog_ch{1,i},window,winDisp*fs,freq_arr,fs);
    testsub1f5_15{i} = mean(abs(s(6:16,:)),1);
    testsub1f20_25{i} = mean(abs(s(21:26,:)),1);
    testsub1f75_115{i} = mean(abs(s(76:116,:)),1);
    testsub1f125_160{i} = mean(abs(s(126:161,:)),1);
    testsub1f160_175{i} = mean(abs(s(161:176,:)),1);
end

% Constructing a new X (R) matrix

v = 54; % 62 channels
N = 3; % time windows 
f = 6; % 6 features
sub1X = ones(5999,v*N*f+1);
sub1XTest = ones(5999,v*N*f+1);


for j = 1:v
    %disp(j);
    for i = N:5999
        % error with sub1f20_25 input
    	sub1X(i,((j-1)*N*f+2):(j*N*f)+1) = [sub1tdv{j}(i-N+1:i) sub1f5_15{j}(i-N+1:i) sub1f20_25{j}(i-N+1:i) ...
            sub1f75_115{j}(i-N+1:i) sub1f125_160{j}(i-N+1:i) sub1f160_175{j}(i-N+1:i)]; %insert data into R
    end
end



sub1X(1:N-1,:) = [];



% Test R
v = 54; % 62 channels
N = 3; % time windows 
f = 6; % 6 features
xLen = 147500;
fs = 1000;
winLen = 100 * 1e-3;
winDisp = 50 * 1e-3;
NumWins = @(xLen, fs, winLen, winDisp) length(0:winDisp*fs:xLen)-(winLen/winDisp);

sub1XTest = ones(NumWins(xLen,fs,winLen,winDisp),v*N*f+1);

for j = 1:v
    %disp(j);
    for i = N:NumWins(xLen,fs,winLen,winDisp)
       
    	sub1XTest(i,((j-1)*N*f+2):(j*N*f)+1) = [sub1tdvTest{j}(i-N+1:i) testsub1f5_15{j}(i-N+1:i) testsub1f20_25{j}(i-N+1:i) ...
            testsub1f75_115{j}(i-N+1:i) testsub1f125_160{j}(i-N+1:i) testsub1f160_175{j}(i-N+1:i)]; %insert data into R
    end
end

sub1XTest(1:2,:) = [];
% Calculation 
sub1fingerflexion = [sub1DataGlove{1} sub1DataGlove{2} sub1DataGlove{3} sub1DataGlove{4} sub1DataGlove{5}];


[B1, FitInfo] = lasso(sub1X,sub1fingerflexion(N:end,1));
lassTestPred1 = sub1XTest*B1 + repmat(FitInfo.Intercept,size((sub1XTest*B1),1),1);

disp('lasso 1 done')
[B2, FitInfo] = lasso(sub1X,sub1fingerflexion(N:end,2));
lassTestPred2 = sub1XTest*B2 + repmat(FitInfo.Intercept,size((sub1XTest*B2),1),1);

disp('lasso 2 done')
[B3, FitInfo] = lasso(sub1X,sub1fingerflexion(N:end,3));
lassTestPred3 = sub1XTest*B3 + repmat(FitInfo.Intercept,size((sub1XTest*B3),1),1);



[B5, FitInfo] = lasso(sub1X,sub1fingerflexion(N:end,5));
lassTestPred5 = sub1XTest*B5 + repmat(FitInfo.Intercept,size((sub1XTest*B5),1),1);


lassTestPred1 = lassTestPred1(:,1);
lassTestPred2 = lassTestPred2(:,1);
lassTestPred3 = lassTestPred3(:,1);
lassTestPred5 = lassTestPred5(:,1);

% spline
sub1Spline1 = spline(50.*(1:length(lassTestPred1)),lassTestPred1',(50:50*length(lassTestPred1)));
sub1Pad1 = [zeros(1,150) sub1Spline1 zeros(1,49)];
sub1Final1 = sub1Pad1';

sub1Spline2 = spline(50.*(1:length(lassTestPred2)),lassTestPred2',(50:50*length(lassTestPred2)));
sub1Pad2 = [zeros(1,150) sub1Spline2 zeros(1,49)];
sub1Final2 = sub1Pad2';

sub1Spline3 = spline(50.*(1:length(lassTestPred3)),lassTestPred3',(50:50*length(lassTestPred3)));
sub1Pad3 = [zeros(1,150) sub1Spline3 zeros(1,49)];
sub1Final3 = sub1Pad3';

sub1Spline5 = spline(50.*(1:length(lassTestPred5)),lassTestPred5',(50:50*length(lassTestPred5)));
sub1Pad5 = [zeros(1,150) sub1Spline5 zeros(1,49)];
sub1Final5 = sub1Pad5';

sub1chp2 = [sub1Final1 sub1Final2 sub1Final3 zeros(147500,1) sub1Final5];
save sub1checkpoint2ft6 sub1chp2