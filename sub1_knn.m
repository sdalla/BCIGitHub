% for this data set, the band-specific ECoG signals were generated using 
% equiripple finite impulse response (FIR) filters by setting their band-pass
% specifications as: sub-bands (1-60 Hz), gamma band (60-100 Hz), and fast 
% gamma band (100-200 Hz)
% testing on sub1 only, for now
%% load data
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
channel = 1:62;
channel(ch_remove) = [];
ind = 1;
for j = channel
    Sub1_Training_ecog_ch{ind} = Sub1_Training_ecog{j};
    Sub1_Test_ecog_ch{ind} = Sub1_Leaderboard_ecog{j};
    ind = ind + 1;
end

v = 62 - length(ch_remove);


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
    sub1_1_60filt{i} = filtfilt(Filter1,1.0,Sub1_Training_ecog_ch{1,i});
    sub1_60_100filt{i} = filtfilt(Filter2,1.0,Sub1_Training_ecog_ch{1,i});
    sub1_100_200filt{i} = filtfilt(Filter3,1.0,Sub1_Training_ecog_ch{1,i});
    sub1_1_60Testfilt{i} = filtfilt(Filter1,1.0,Sub1_Test_ecog_ch{1,i});
    sub1_60_100Testfilt{i} = filtfilt(Filter2,1.0,Sub1_Test_ecog_ch{1,i});
    sub1_100_200Testfilt{i} = filtfilt(Filter3,1.0,Sub1_Test_ecog_ch{1,i});
end

%% Amplitude modulation
ampFxn = @(x) sum(x.^2);
for i = 1:v
    sub1_1_60amp{i} = MovingWinFeats(sub1_1_60filt{i}, fs, winLen, winDisp, ampFxn);
    sub1_60_100amp{i} = MovingWinFeats(sub1_60_100filt{i}, fs, winLen, winDisp, ampFxn);
    sub1_100_200amp{i} = MovingWinFeats(sub1_100_200filt{i}, fs, winLen, winDisp, ampFxn);
    sub1_1_60Testamp{i} = MovingWinFeats(sub1_1_60Testfilt{i}, fs, winLen, winDisp, ampFxn);
    sub1_60_100Testamp{i} = MovingWinFeats(sub1_60_100Testfilt{i}, fs, winLen, winDisp, ampFxn);
    sub1_100_200Testamp{i} = MovingWinFeats(sub1_100_200Testfilt{i}, fs, winLen, winDisp, ampFxn);
end

%% Constructing a new X (R) matrix
N = 4; % time windows 
f = 3; % 6 features
sub1X = ones(5999,v*N*f+1);
sub1XTest = ones(5999,v*N*f+1);

for j = 1:v
    for i = N:5999
        sub1X(i,((j-1)*N*f+2):(j*N*f)+1) = [sub1_1_60amp{j}(i-N+1:i) sub1_60_100amp{j}(i-N+1:i) ...
            sub1_100_200amp{j}(i-N+1:i)]; %insert data into R
    end
end

sub1X(1:N-1,:) = [];
sub1XTest(1:N-1,:) = [];

%% Test R
xLen = 147500;
fs = 1000;
winLen = 100 * 1e-3;
winDisp = 50 * 1e-3;
NumWins = @(xLen, fs, winLen, winDisp) length(0:winDisp*fs:xLen)-(winLen/winDisp);

sub1XTest = ones(NumWins(xLen,fs,winLen,winDisp),v*N*f+1);

for j = 1:v
    for i = N:NumWins(xLen,fs,winLen,winDisp)
        sub1XTest(i,((j-1)*N*f+2):(j*N*f)+1) = [sub1_1_60Testamp{j}(i-N+1:i) sub1_60_100Testamp{j}(i-N+1:i) ...
            sub1_100_200Testamp{j}(i-N+1:i)]; %insert data into R
    end
end

sub1XTest(1:N-1,:) = [];
%% Calculation 
sub1fingerflexion = [sub1DataGlove{1} sub1DataGlove{2} sub1DataGlove{3} sub1DataGlove{4} sub1DataGlove{5}];

[B1, FitInfo1] = lasso(sub1X,sub1fingerflexion(N:end,1));
lassTestPred1 = sub1XTest*B1 + repmat(FitInfo1.Intercept,size((sub1XTest*B1),1),1);

disp('lasso 1 done')
[B2, FitInfo2] = lasso(sub1X,sub1fingerflexion(N:end,2));
lassTestPred2 = sub1XTest*B2 + repmat(FitInfo2.Intercept,size((sub1XTest*B2),1),1);

disp('lasso 2 done')
[B3, FitInfo3] = lasso(sub1X,sub1fingerflexion(N:end,3));
lassTestPred3 = sub1XTest*B3 + repmat(FitInfo3.Intercept,size((sub1XTest*B3),1),1);

[B5, FitInfo5] = lasso(sub1X,sub1fingerflexion(N:end,5));
lassTestPred5 = sub1XTest*B5 + repmat(FitInfo5.Intercept,size((sub1XTest*B5),1),1);

lassTestPred1 = lassTestPred1(:,1);

lassTestPred2 = lassTestPred2(:,1);

lassTestPred3 = lassTestPred3(:,1);

lassTestPred5 = lassTestPred5(:,1);



%% spline
sub1Spline1 = spline(50.*(1:length(lassTestPred1)),lassTestPred1',(50:50*length(lassTestPred1)));
sub1Pad1 = [zeros(1,200) sub1Spline1 zeros(1,49)];
sub1Final1 = sub1Pad1';


sub1Spline2 = spline(50.*(1:length(lassTestPred2)),lassTestPred2',(50:50*length(lassTestPred2)));
sub1Pad2 = [zeros(1,200) sub1Spline2 zeros(1,49)];
sub1Final2 = sub1Pad2';



sub1Spline3 = spline(50.*(1:length(lassTestPred3)),lassTestPred3',(50:50*length(lassTestPred3)));
sub1Pad3 = [zeros(1,200) sub1Spline3 zeros(1,49)];
sub1Final3 = sub1Pad3';



sub1Spline5 = spline(50.*(1:length(lassTestPred5)),lassTestPred5',(50:50*length(lassTestPred5)));
sub1Pad5 = [zeros(1,200) sub1Spline5 zeros(1,49)];
sub1Final5 = sub1Pad5';

%% filtering w medfilt
sub1Final1 = medfilt1(sub1Final1(:,1),1000);
sub1Final2 = medfilt1(sub1Final2(:,1),1000);
sub1Final3 = medfilt1(sub1Final3(:,1),1000);
sub1Final5 = medfilt1(sub1Final5(:,1),1000);
sub1chp2 = [sub1Final1 sub1Final2 sub1Final3 zeros(147500,1) sub1Final5];

%% knn
clc;
clear all;
load('sub1checkpoint2a.mat');
load('Sub1_Training_dg.mat');
labelMatrix = NaN(size(Sub1_Training_dg{1},1),5);
sub1knn = NaN(size(sub1chp2,1),5);

for j = 1:5
    for i = 1:size(Sub1_Training_dg{1},1)
        if Sub1_Training_dg{j}(i) > .5
            labelMatrix(i,j) = 1;
        else labelMatrix(i,j) = 0;
        end
    end
    knnmodel = knnclassify(sub1chp2(:,j),Sub1_Training_dg{j},labelMatrix(:,j));
    sub1knn(:,j) = sub1chp2(:,j).*knnmodel;
end


%% saving
save sub1knn sub1knn

%% 
%% knn sub 2
clc;
clear all;
load('sub2checkpoint2a.mat');
load('Sub2_Training_dg.mat');
labelMatrix = NaN(size(Sub2_Training_dg{1},1),5);
sub2knn = NaN(size(sub2chp2,1),5);

for j = 1:5
    for i = 1:size(Sub2_Training_dg{1},1)
        if Sub2_Training_dg{j}(i) > .5
            labelMatrix(i,j) = 1;
        else labelMatrix(i,j) = 0;
        end
    end
    knnmodel = knnclassify(sub2chp2(:,j),Sub2_Training_dg{j},labelMatrix(:,j));
    sub2knn(:,j) = sub2chp2(:,j).*knnmodel;
end


%% saving
save sub2knn sub2knn

%% knn sub 3
clc;
clear all;
load('sub3checkpoint2a.mat');
load('Sub3_Training_dg.mat');
labelMatrix = NaN(size(Sub3_Training_dg{1},1),5);
sub3knn = NaN(size(sub3chp2,1),5);

for j = 1:5
    for i = 1:size(Sub3_Training_dg{1},1)
        if Sub3_Training_dg{j}(i) > .5
            labelMatrix(i,j) = 1;
        else labelMatrix(i,j) = 0;
        end
    end
    knnmodel = knnclassify(sub3chp2(:,j),Sub3_Training_dg{j},labelMatrix(:,j));
    sub3knn(:,j) = sub3chp2(:,j).*knnmodel;
end


%% saving
save sub3knn sub3knn