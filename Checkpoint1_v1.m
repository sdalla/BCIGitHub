
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
   sub1tdv{i} = MovingWinFeats(Sub1_training_ecog{1,i,1}, fs, winLen, winDisp, tdvFxn);
end
%% Feature Extraction (Average Frequency-Domain Magnitude in 5 bands)
% Frequency bands are: 5-15Hz, 20-25Hz, 75-115Hz, 125-160Hz, 160-175Hz
% Total number of features in given time window is (num channels)*(5+1)
window = winLen*fs;

%subject 1
for i = 1:62
    [s,f,t] = spectrogram(Sub1_training_ecog{1,1,1},window,winDisp*fs,[],fs);
    sub1f5_15{i} = mean(s(f(f>5)<15,:),1);
    sub1f20_25{i} = mean(s(f(f>20)<25,:),1);
    sub1f75_115{i} = mean(s(f(f>75)<115,:),1);
    sub1f125_160{i} = mean(s(f(f>125)<160,:),1);
    sub1f160_175{i} = mean(s(f(f>160)<175,:),1);
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
    for i = 1:5999-N+1
        % error with sub1f20_25 input
    	sub1X(i,((j-1)*N*f+2):(j*N*f)+1) = [sub1tdv{j}(i:(i+N-1)) sub1f5_15{j}(i:(i+N-1)) sub1f20_25{j}(i:(i+N-1)) ...
            sub1f75_115{j}(i:(i+N-1)) sub1f125_160{j}(i:(i+N-1)) sub1f160_175{j}(i:(i+N-1))]; %insert data into R
    end
end

%% Calculation
f = zeros(62*N+1,2);
f(:,1) = mldivide(mldivide(X,X),mldivide(R,s(:,1)));
f(:,2) = mldivide(mldivide(X,X),mldivide(R,s(:,2)));

    
    
    %% Calculation - Shira
sub1fingerflexion = [sub1DataGlove{1} sub1DataGlove{2} sub1DataGlove{3} sub1DataGlove{4} sub1DataGlove{5}];
sub1_weight = zeros(62*N+1,5);
sub1X = real(sub1X);
%sub1_weight = mldivide(mldivide(sub1X,sub1X),mldivide(R,sub1fingerflexion));
sub1_weight = mldivide((sub1X.'*sub1X),(sub1X.'*sub1fingerflexion));
sub1_predict = sub1X*sub1_weight;


<<<<<<< HEAD

%% spline stuff

interp = 1/fs:1/fs:300;
time = 1/fs:50/fs:300-50/fs;
sub1Spline = spline(time,sub1_predict(:,1),interp);


=======
%% spline stuff


time = 50/fs:50/fs:300;
% the first 50ms we dont have predictions for, will zero pad at the end
interp = linspace(time(1),time(end),300000-0.05*fs);
sub1Spline = spline(time,ypred,interp);
>>>>>>> c20ddcd7173afe8531d26f61d041faa170333926
