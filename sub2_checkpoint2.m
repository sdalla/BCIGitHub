% for this data set, the band-specific ECoG signals were generated using 
% equiripple finite impulse response (FIR) filters by setting their band-pass
% specifications as: sub-bands (1-60 Hz), gamma band (60-100 Hz), and fast 
% gamma band (100-200 Hz)
% testing on sub2 only, for now
%% load data
xLen = 300000;
fs = 1000;
winLen = 100 * 1e-3;
winDisp = 50 * 1e-3;
NumWins = @(xLen, fs, winLen, winDisp) length(0:winDisp*fs:xLen)-(winLen/winDisp);
% ecog data
load('Sub2_training_ecog.mat');
% original dataglove
load('sub2_Training_dg.mat');
% decimated dataglove
%load('Sub2DataGlove.mat');
%leaderboard ecog
load('Sub2_Leaderboard_ecog.mat');

ch_remove = [23 37 38]; %remove channels found in other mat file
channel = 1:45;
channel(ch_remove) = [];
ind = 1;
for j = channel
    Sub2_Training_ecog_ch{ind} = Sub2_Training_ecog{j};
    Sub2_Test_ecog_ch{ind} = Sub2_Leaderboard_ecog{j};
    ind = ind + 1;
end
%% break into freq bands using spectrogram, could do next section instead
v = size(Sub2_Training_ecog_ch);
v = v(1,2);
window = winLen*fs;
freq_arr = 0:1:500; 
%subject 1
for i = 1:v
    [s,freq,t] = spectrogram(Sub2_Training_ecog_ch{1,i},window,winDisp*fs,freq_arr,fs);
    sub2f1_60{i} = abs(s(1:60,:));
    sub2f60_100{i} = abs(s(60:100,:));
    sub2f100_200{i} = abs(s(100:200,:));
    [s,freq,t] = spectrogram(Sub2_Test_ecog_ch{1,i},window,winDisp*fs,freq_arr,fs);
    sub2f1_60Test{i} = abs(s(1:60,:));
    sub2f60_100Test{i} = abs(s(60:100,:));
    sub2f100_200Test{i} = abs(s(100:200,:));
    
end

%% break into freq bands using FIR filters and filterDesigner

% filter1 has passband cutoffs of 1 and 60Hz, stopband db of 50 pass of 1
% stopsbands are 0.5Hz above/below 

temp = load('Filter1.mat');
Filter1 = temp.Filter1;
temp = load('Filter2.mat');
Filter2 = temp.Filter2;
temp = load('Filter3.mat');
Filter3 = temp.Filter3;

for i = 1:v   
    sub2_1_60filt{i} = filtfilt(Filter1,1.0,Sub2_Training_ecog_ch{1,i});
    sub2_60_100filt{i} = filtfilt(Filter2,1.0,Sub2_Training_ecog_ch{1,i});
    sub2_100_200filt{i} = filtfilt(Filter3,1.0,Sub2_Training_ecog_ch{1,i});
    sub2_1_60Testfilt{i} = filtfilt(Filter1,1.0,Sub2_Test_ecog_ch{1,i});
    sub2_60_100Testfilt{i} = filtfilt(Filter2,1.0,Sub2_Test_ecog_ch{1,i});
    sub2_100_200Testfilt{i} = filtfilt(Filter3,1.0,Sub2_Test_ecog_ch{1,i});
end


%% Amplitude modulation
ampFxn = @(x) sum(x.^2);
for i = 1:v
    sub2_1_60amp{i} = MovingWinFeats(sub2_1_60filt{i}, fs, winLen, winDisp, ampFxn);
    sub2_60_100amp{i} = MovingWinFeats(sub2_60_100filt{i}, fs, winLen, winDisp, ampFxn);
    sub2_100_200amp{i} = MovingWinFeats(sub2_100_200filt{i}, fs, winLen, winDisp, ampFxn);
    sub2_1_60Testamp{i} = MovingWinFeats(sub2_1_60Testfilt{i}, fs, winLen, winDisp, ampFxn);
    sub2_60_100Testamp{i} = MovingWinFeats(sub2_60_100Testfilt{i}, fs, winLen, winDisp, ampFxn);
    sub2_100_200Testamp{i} = MovingWinFeats(sub2_100_200Testfilt{i}, fs, winLen, winDisp, ampFxn);
end

%% Constructing a new X (R) matrix

v = 42; % 62 channels
N = 4; % time windows 
f = 3; % 6 features
sub2X = ones(5999,v*N*f+1);
sub2XTest = ones(5999,v*N*f+1);


for j = 1:v
  
    for i = N:5999
     
        sub2X(i,((j-1)*N*f+2):(j*N*f)+1) = [sub2_1_60amp{j}(i-N+1:i) sub2_60_100amp{j}(i-N+1:i) ...
            sub2_100_200amp{j}(i-N+1:i)]; %insert data into R
    end
end

sub2X(1:N-1,:) = [];
sub2XTest(1:N-1,:) = [];


%% Test R
v = 42; % 62 channels
N = 4; % time windows 
f = 3; % 6 features
xLen = 147500;
fs = 1000;
winLen = 100 * 1e-3;
winDisp = 50 * 1e-3;
NumWins = @(xLen, fs, winLen, winDisp) length(0:winDisp*fs:xLen)-(winLen/winDisp);

sub2XTest = ones(NumWins(xLen,fs,winLen,winDisp),v*N*f+1);

for j = 1:v
    %disp(j);
    for i = N:NumWins(xLen,fs,winLen,winDisp)
       
    	        sub2XTest(i,((j-1)*N*f+2):(j*N*f)+1) = [sub2_1_60Testamp{j}(i-N+1:i) sub2_60_100Testamp{j}(i-N+1:i) ...
            sub2_100_200Testamp{j}(i-N+1:i)]; %insert data into R

    end
end

sub2XTest(1:N-1,:) = [];
%% Calculation 
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

%% spline
sub2Spline1 = spline(50.*(1:length(lassTestPred1)),lassTestPred1',(50:50*length(lassTestPred1)));
sub2Pad1 = [zeros(1,200) sub2Spline1 zeros(1,49)];
sub2Final1 = sub2Pad1';

sub2Spline2 = spline(50.*(1:length(lassTestPred2)),lassTestPred2',(50:50*length(lassTestPred2)));
sub2Pad2 = [zeros(1,200) sub2Spline2 zeros(1,49)];
sub2Final2 = sub2Pad2';

sub2Spline3 = spline(50.*(1:length(lassTestPred3)),lassTestPred3',(50:50*length(lassTestPred3)));
sub2Pad3 = [zeros(1,200) sub2Spline3 zeros(1,49)];
sub2Final3 = sub2Pad3';

sub2Spline5 = spline(50.*(1:length(lassTestPred5)),lassTestPred5',(50:50*length(lassTestPred5)));
sub2Pad5 = [zeros(1,200) sub2Spline5 zeros(1,49)];
sub2Final5 = sub2Pad5';
%%
sub2chp2 = [sub2Final1 sub2Final2 sub2Final3 zeros(147500,1) sub2Final5];
save sub2checkpoint2 sub2chp2