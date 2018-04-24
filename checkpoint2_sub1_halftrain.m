load('Sub1_Training_ecog.mat');
%% Identifying channels with Excessive Line Noise
%sampling freq
Fs = 1000;
%length of the sample
l = length(Sub1_Training_ecog{1});
%freq array of sample
f = Fs*(0:(l/2))/l;
ind = 1;
for i = 1:62
    if i == 55
        continue
    end
fft_sub1 = fft(Sub1_Training_ecog{i});
fft_mag = abs((fft_sub1));
fft_mag_plot = fft_mag(1:l/2+1); 

noise(ind) = mean(fft_mag(find(f==60)-75:find(f==60)+75));
ind = ind + 1;
end
figure()
plot(noise/median(noise),'o')

%channel numbers that have line noises that have statistically significant
%deviations
abnormalAmpCh = find(noise/median(noise) > mean(noise/median(noise)) + 2*std(noise/median(noise)));

%% Identify channels with Abnormal Amplitude Distributions
% Log-transform the RMS of the raw voltage values (RMSVs) for all channels 
data = Sub1_Training_ecog;
data(55) = [];
RMSV = cellfun(@rms,data);
logRMSV = log10(RMSV);

% Normalize  by calculating z-scores
zscores = zscore(logRMSV);
scatter(1:61,zscores);

% Calculate p-values using a Gaussian distribution fitted to the data
[h,p] = ttest(logRMSV);
channelsExcl = find(logRMSV > (mean(logRMSV)+2*std(logRMSV)));

% Exclude channel if mean amplitude is significantly different from the mean amplitude of all other channels (p < 0.05). 
%% Numwins
NumWins = @(xLen, fs, winLen, winDisp) length(0:winDisp*fs:xLen)-(winLen/winDisp);

%% Feature Extraction (Average Time-Domain Voltage)

tdvFxn = @(x) mean(x);

xLen = 300000;
fs = 1000;
winLen = .1;
winDisp = .05;

%subject 1
sub1tdv = cell(1,62);
ind = 1;
for i = 1:62
   % if (i == 55) || (i == 21) || (i == 44) || (i == 52)
     %   continue
   % end
   sub1tdv{i} = MovingWinFeats(Sub1_Training_ecog{1,i}, fs, winLen, winDisp, tdvFxn);
   %ind = ind+1;
end
%% Feature Extraction (Average Frequency-Domain Magnitude in 5 bands)
% Frequency bands are: 5-15Hz, 20-25Hz, 75-115Hz, 125-160Hz, 160-175Hz
% Total number of features in given time window is (num channels)*(5+1)
window = winLen*fs;
freq_arr = 0:1:1000; %change to 0 to 1000 & change indices below
%subject 1
ind = 1;
for i = 1:62
   % if (i == 55) || (i == 21) || (i == 44) || (i == 52)
        %continue
   % end
    [s,freq,t] = spectrogram(Sub1_Training_ecog{1,i},window,winDisp*fs,1000,fs);
    sub1f5_15{i} = mean(abs(s(6:16,:)),1);
    sub1f20_25{i} = mean(abs(s(21:26,:)),1);
    sub1f75_115{i} = mean(abs(s(76:116,:)),1);
    sub1f125_160{i} = mean(abs(s(126:161,:)),1);
    sub1f160_175{i} = mean(abs(s(161:176,:)),1);
    
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
v = 62-4; % 62 channels
N = 3; % 3 time windows 
f = 6; % 6 features
sub1X = ones(5999,v*N*f+1);
ind = 1;
for j = 1:v
    %disp(j);
    if (i == 55) || (i == 21) || (i == 44) || (i == 52) || (i == 18) || (i == 27) || (i == 40) || (i==49) %remove outlier (channel 55)
        continue
    end
    for i = N:5999
       
        % error with sub1f20_25 input
    	sub1X(i,((ind-1)*N*f+2):(ind*N*f)+1) = [sub1tdv{j}(i-N+1:i) sub1f5_15{j}(i-N+1:i) sub1f20_25{j}(i-N+1:i) ...
            sub1f75_115{j}(i-N+1:i) sub1f125_160{j}(i-N+1:i) sub1f160_175{j}(i-N+1:i)]; %insert data into R
    
    end
    ind = ind +1;
end

sub1X(1:2,:) = [];
    
%% Split into test and train
%sub1X = abs(sub1X);
sub1X_train = sub1X(1:3000,:);
sub1X_test = sub1X(3001:end,:);
%% Calculation 
sub1fingerflexion = [sub1DataGlove{1} sub1DataGlove{2} sub1DataGlove{3} sub1DataGlove{4} sub1DataGlove{5}];
sub1fingerflexion_train = sub1fingerflexion(N:3000+N-1,:);
sub1fingerflexion_test = sub1fingerflexion(3000+N:end,:);

arg1 = (sub1X_train'*sub1X_train);
arg2 = (sub1X_train'*sub1fingerflexion_train);
sub1_weight = mldivide(arg1,arg2);
sub1_trainpredict = sub1X_train*sub1_weight;

sub1_testpredict = sub1X_test*sub1_weight;
testcorr = mean(diag(corr(sub1_testpredict, sub1fingerflexion_test)))

%% Prediction Using Lasso
arg1 = sub1X_train;
[B1, FitInfo] = lasso(arg1,sub1fingerflexion_train(:,1));

[B1, FitInfo] = lasso(sub1X_train,sub1fingerflexion_train(:,1));

lassTestPredx = sub1X_test*B1 + repmat(FitInfo.Intercept,size((sub1X_test*B1),1),1);
lassocorr = mean(corr(lassTestPredx, sub1fingerflexion_test(:,1)))

% %% spline stuff
% % will zero pad at the end
% % [1:lastSample].*50 to reconstruct as much as we can then pad to 150k pt
% % sub1_predict is our prediction on our testing data
% % which will be 50th sample to the 2947*50th sample
%sub1Spline = spline(50.*(1:length(sub1_testpredict)),sub1_testpredict',(50:50*length(sub1_testpredict)));
% % remember to un-transpose sub1_testpredict at the end
%sub1Pad = padarray(sub1Spline, [0 99]);
%sub1Pad(:,end+1) = 0;
%sub1Final = sub1Pad';
