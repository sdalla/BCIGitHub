%% load data
xLen = 300000;
fs = 1000;
winLen = 100 * 1e-3;
winDisp = 50 * 1e-3;
NumWins = @(xLen, fs, winLen, winDisp) length(0:winDisp*fs:xLen)-(winLen/winDisp);
% ecog data
load('Sub3_training_ecog.mat');
% original dataglove
load('sub3_Training_dg.mat');
% decimated dataglove
%% Decimation of dataglove
load('Sub3_Training_dg.mat');
% decimated glove data for subject one
% take out the last value to match our 5999
sub3DataGlove = cell(1,5);
for i = 1:5
    sub3DataGlove{i} = decimate(Sub3_Training_dg{i},50);
    sub3DataGlove{i}(end)= [];
end
load('sub3DataGlove.mat');
%leaderboard ecog
load('Sub3_Leaderboard_ecog.mat');

ch_remove = [ 23 38 58]; %remove channels found in other mat file
channel = 1:64;
channel(ch_remove) = [];
ind = 1;
for j = channel
    Sub3_Training_ecog_ch{ind} = Sub3_Training_ecog{j};
    Sub3_Test_ecog_ch{ind} = Sub3_Leaderboard_ecog{j};
    ind = ind + 1;
end

%% break into freq bands using FIR filters and filterDesigner
v = 64 - length(ch_remove);
window = winLen*fs;
freq_arr = 0:1:500; 
% filter1 has passband cutoffs of 1 and 60Hz, stopband db of 50 pass of 1
% stopsbands are 0.5Hz above/below 
temp = load('Filter1.mat');
Filter1 = temp.Filter1;
temp = load('Filter2.mat');
Filter2 = temp.Filter2;
temp = load('Filter3.mat');
Filter3 = temp.Filter3;

for i = 1:v   
    sub3_1_60filt{i} = filtfilt(Filter1,1.0,Sub3_Training_ecog_ch{1,i});
    sub3_60_100filt{i} = filtfilt(Filter2,1.0,Sub3_Training_ecog_ch{1,i});
    sub3_100_200filt{i} = filtfilt(Filter3,1.0,Sub3_Training_ecog_ch{1,i});
    sub3_1_60Testfilt{i} = filtfilt(Filter1,1.0,Sub3_Test_ecog_ch{1,i});
    sub3_60_100Testfilt{i} = filtfilt(Filter2,1.0,Sub3_Test_ecog_ch{1,i});
    sub3_100_200Testfilt{i} = filtfilt(Filter3,1.0,Sub3_Test_ecog_ch{1,i});
end

%% Amplitude modulation
ampFxn = @(x) sum(x.^2);
for i = 1:v
    sub3_1_60amp{i} = MovingWinFeats(sub3_1_60filt{i}, fs, winLen, winDisp, ampFxn);
    sub3_60_100amp{i} = MovingWinFeats(sub3_60_100filt{i}, fs, winLen, winDisp, ampFxn);
    sub3_100_200amp{i} = MovingWinFeats(sub3_100_200filt{i}, fs, winLen, winDisp, ampFxn);
    sub3_1_60Testamp{i} = MovingWinFeats(sub3_1_60Testfilt{i}, fs, winLen, winDisp, ampFxn);
    sub3_60_100Testamp{i} = MovingWinFeats(sub3_60_100Testfilt{i}, fs, winLen, winDisp, ampFxn);
    sub3_100_200Testamp{i} = MovingWinFeats(sub3_100_200Testfilt{i}, fs, winLen, winDisp, ampFxn);
end

%% Constructing a new X (R) matrix
N = 4; % time windows 
f = 3; % 6 features
sub3X = ones(5999,v*N*f+1);
sub3XTest = ones(5999,v*N*f+1);

for j = 1:v
    for i = N:5999
        sub3X(i,((j-1)*N*f+2):(j*N*f)+1) = [sub3_1_60amp{j}(i-N+1:i) sub3_60_100amp{j}(i-N+1:i) ...
            sub3_100_200amp{j}(i-N+1:i)]; %insert data into R
    end
end

sub3X(1:N-1,:) = [];

%% Test R
xLen = 147500;
fs = 1000;
winLen = 100 * 1e-3;
winDisp = 50 * 1e-3;
NumWins = @(xLen, fs, winLen, winDisp) length(0:winDisp*fs:xLen)-(winLen/winDisp);

sub3XTest = ones(NumWins(xLen,fs,winLen,winDisp),v*N*f+1);

for j = 1:v
    %disp(j);
    for i = N:NumWins(xLen,fs,winLen,winDisp)
        sub3XTest(i,((j-1)*N*f+2):(j*N*f)+1) = [sub3_1_60Testamp{j}(i-N+1:i) sub3_60_100Testamp{j}(i-N+1:i) ...
            sub3_100_200Testamp{j}(i-N+1:i)]; %insert data into R
    end
end

sub3XTest(1:N-1,:) = [];
%% Calculation 
sub3fingerflexion = [sub3DataGlove{1} sub3DataGlove{2} sub3DataGlove{3} sub3DataGlove{4} sub3DataGlove{5}];

[B1, FitInfo1] = lasso(sub3X,sub3fingerflexion(N:end,1));
lassTestPred1 = sub3XTest*B1 + repmat(FitInfo1.Intercept,size((sub3XTest*B1),1),1);

disp('lasso 1 done')
[B2, FitInfo2] = lasso(sub3X,sub3fingerflexion(N:end,2));
lassTestPred2 = sub3XTest*B2 + repmat(FitInfo2.Intercept,size((sub3XTest*B2),1),1);

disp('lasso 2 done')
[B3, FitInfo3] = lasso(sub3X,sub3fingerflexion(N:end,3));
lassTestPred3 = sub3XTest*B3 + repmat(FitInfo3.Intercept,size((sub3XTest*B3),1),1);

disp('lasso 3 done')
[B5, FitInfo5] = lasso(sub3X,sub3fingerflexion(N:end,5));
lassTestPred5 = sub3XTest*B5 + repmat(FitInfo5.Intercept,size((sub3XTest*B5),1),1);

lassTestPred1 = lassTestPred1(:,1);
lassTestPred2 = lassTestPred2(:,1);
lassTestPred3 = lassTestPred3(:,1);
lassTestPred5 = lassTestPred5(:,1);

%% spline
sub3Spline1 = spline(50.*(1:length(lassTestPred1)),lassTestPred1',(50:50*length(lassTestPred1)));
sub3Pad1 = [zeros(1,200) sub3Spline1 zeros(1,49)];
sub3Final1 = sub3Pad1';

sub3Spline2 = spline(50.*(1:length(lassTestPred2)),lassTestPred2',(50:50*length(lassTestPred2)));
sub3Pad2 = [zeros(1,200) sub3Spline2 zeros(1,49)];
sub3Final2 = sub3Pad2';

sub3Spline3 = spline(50.*(1:length(lassTestPred3)),lassTestPred3',(50:50*length(lassTestPred3)));
sub3Pad3 = [zeros(1,200) sub3Spline3 zeros(1,49)];
sub3Final3 = sub3Pad3';

sub3Spline5 = spline(50.*(1:length(lassTestPred5)),lassTestPred5',(50:50*length(lassTestPred5)));
sub3Pad5 = [zeros(1,200) sub3Spline5 zeros(1,49)];
sub3Final5 = sub3Pad5';


%% post processing of the output (filtering w filter design)

%sampling freq
Fs = 1000;
%length of the sample
l = length(sub3chp2(:,1));
%freq array of sample
f = Fs*(0:(l/2))/l;

fft_sub1 = fft(sub3chp2(:,1));
fft_mag = abs((fft_sub1));
fft_mag_plot = fft_mag(1:l/2+1); 
figure()
plot(f,fft_mag_plot)
figure()
plot(sub3chp2(:,1))
temp = load('postFilter.mat');
postFilter = temp.postFilter;

filteredOutput = filtfilt(postFilter,1.0,sub3chp2(:,1));
figure()
plot(filteredOutput)
%% filtering with  medfilt
sub3Final1 = medfilt1(sub3Final1(:,1),1000);
sub3Final2 = medfilt1(sub3Final2(:,1),1000);
sub3Final3 = medfilt1(sub3Final3(:,1),1000);
sub3Final5 = medfilt1(sub3Final5(:,1),1000);

%% saving the data to a .mat file
sub3chp2 = [sub3Final1 sub3Final2 sub3Final3 zeros(147500,1) sub3Final5];
save sub3checkpoint2a sub3chp2