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
% ecog data
load('Sub1_training_ecog.mat');
% original dataglove
load('sub1_Training_dg.mat');
% decimated dataglove
load('sub1DataGlove.mat');

i = [55 21 44 52 18 27 40 49]; %remove channels
for j = 1:length(i)
Sub1_Training_ecog(i(j))=[];
end
%% break into freq bands using spectrogram, could do next section instead
v = 54;
window = winLen*fs;
freq_arr = 0:1:500; 
%subject 1
for i = 1:v
    [s,freq,t] = spectrogram(Sub1_Training_ecog{1,i},window,winDisp*fs,freq_arr,fs);
    sub1f1_60{i} = abs(s(1:60,:));
    sub1f60_100{i} = abs(s(60:100,:));
    sub1f100_200{i} = abs(s(100:200,:));
    
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
    sub1_1_60filt{i} = filtfilt(Filter1,1.0,Sub1_Training_ecog{1,i});
    sub1_60_100filt{i} = filtfilt(Filter2,1.0,Sub1_Training_ecog{1,i});
    sub1_100_200filt{i} = filtfilt(Filter3,1.0,Sub1_Training_ecog{1,i});
end

%% plot the filtered and original data (sanity check)
subplot(4,1,1)
plot(Sub1_Training_ecog{1,1}(1:10000))
subplot(4,1,2)
plot(sub1_1_60filt{1}(1:10000))
subplot(4,1,3)
plot(sub1_60_100filt{1}(1:10000))
subplot(4,1,4)
plot(sub1_100_200filt{1}(1:10000))

%% Amplitude modulation
ampFxn = @(x) sum(x.^2);
for i = 1:v
    sub1_1_60amp{i} = MovingWinFeats(sub1_1_60filt{i}, fs, winLen, winDisp, ampFxn);
    sub1_60_100amp{i} = MovingWinFeats(sub1_60_100filt{i}, fs, winLen, winDisp, ampFxn);
    sub1_100_200amp{i} = MovingWinFeats(sub1_100_200filt{i}, fs, winLen, winDisp, ampFxn);
end

%% Constructing a new X (R) matrix

v = 54; % 62 channels
N = 4; % time windows 
f = 3; % 6 features
sub1X = ones(5999,v*N*f+1);


for j = 1:v
  
    for i = N:5999
     
        sub1X(i,((j-1)*N*f+2):(j*N*f)+1) = [sub1_1_60amp{j}(i-N+1:i) sub1_60_100amp{j}(i-N+1:i) ...
            sub1_100_200amp{j}(i-N+1:i)]; %insert data into R
    end
end

sub1X(1:N-1,:) = [];

%% Calculation 
sub1fingerflexion = [sub1DataGlove{1} sub1DataGlove{2} sub1DataGlove{3} sub1DataGlove{4} sub1DataGlove{5}];

arg1 = (sub1X'*sub1X);
arg2 = (sub1X'*sub1fingerflexion(N:end,:));
sub1_weight = mldivide(arg1,arg2);
sub1_trainpredict = sub1X*sub1_weight;
corr(sub1_trainpredict,sub1fingerflexion(N:end,:))

%% split into test and train and do the same thing
%% Split into test and train

sub1X_train = sub1X(1:3000,:);
sub1X_test = sub1X(3001:end,:);
%% Calculation 
sub1fingerflexion_train = sub1fingerflexion(N:3000+N-1,:);
sub1fingerflexion_test = sub1fingerflexion(3000+N:end,:);


arg1 = (sub1X_train'*sub1X_train);
arg2 = (sub1X_train'*sub1fingerflexion_train);
sub1_weight = mldivide(arg1,arg2);
sub1_trainpredict = sub1X_train*sub1_weight;

[B1, FitInfo] = lasso(sub1X_train,sub1fingerflexion_train(:,1));
lassTestPredx = sub1X_test*B1 + repmat(FitInfo.Intercept,size((sub1X_test*B1),1),1);
lassocorr = mean(corr(lassTestPredx, sub1fingerflexion_test(:,1)))
%%
%sub1_testpredict = sub1X_test*sub1_weight;
%traingcorr = diag(corr(sub1_trainpredict, sub1fingerflexion_train))
%testcorr = diag(corr(sub1_testpredict, sub1fingerflexion_test))