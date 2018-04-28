%% Amplitude Modulation
ampFxn = @(x) sum(x.^2);

for i = 1:v1
    sub1_1_60Testamp{i} = MovingWinFeats(sub3_1_60Testfilt{i}, fs, winLen, winDisp, ampFxn);
    sub1_60_100Testamp{i} = MovingWinFeats(sub3_60_100Testfilt{i}, fs, winLen, winDisp, ampFxn);
    sub1_100_200Testamp{i} = MovingWinFeats(sub3_100_200Testfilt{i}, fs, winLen, winDisp, ampFxn);
end

for i = 1:v2
    sub2_1_60Testamp{i} = MovingWinFeats(sub3_1_60Testfilt{i}, fs, winLen, winDisp, ampFxn);
    sub2_60_100Testamp{i} = MovingWinFeats(sub3_60_100Testfilt{i}, fs, winLen, winDisp, ampFxn);
    sub2_100_200Testamp{i} = MovingWinFeats(sub3_100_200Testfilt{i}, fs, winLen, winDisp, ampFxn);
end

for i = 1:v3
    sub3_1_60Testamp{i} = MovingWinFeats(sub3_1_60Testfilt{i}, fs, winLen, winDisp, ampFxn);
    sub3_60_100Testamp{i} = MovingWinFeats(sub3_60_100Testfilt{i}, fs, winLen, winDisp, ampFxn);
    sub3_100_200Testamp{i} = MovingWinFeats(sub3_100_200Testfilt{i}, fs, winLen, winDisp, ampFxn);
end
%% Feature Extraction
N = 4; % time windows 
f = 3; % 6 features
xLen = 147500;
fs = 1000;
winLen = 100 * 1e-3;
winDisp = 50 * 1e-3;
NumWins = @(xLen, fs, winLen, winDisp) length(0:winDisp*fs:xLen)-(winLen/winDisp);

% sub 1
sub1XTest = ones(NumWins(xLen,fs,winLen,winDisp),v1*N*f+1);

for j = 1:v1
    disp(j);
    for i = N:NumWins(xLen,fs,winLen,winDisp)
        sub1XTest(i,((j-1)*N*f+2):(j*N*f)+1) = [sub1_1_60Testamp{j}(i-N+1:i) sub1_60_100Testamp{j}(i-N+1:i) ...
            sub1_100_200Testamp{j}(i-N+1:i)]; %insert data into R
    end
end

sub1XTest(1:N-1,:) = [];

% sub 2 f
sub2XTest = ones(NumWins(xLen,fs,winLen,winDisp),v2*N*f+1);

for j = 1:v2
    for i = N:NumWins(xLen,fs,winLen,winDisp)
        sub2XTest(i,((j-1)*N*f+2):(j*N*f)+1) = [sub2_1_60Testamp{j}(i-N+1:i) sub2_60_100Testamp{j}(i-N+1:i) ...
            sub2_100_200Testamp{j}(i-N+1:i)]; %insert data into R
    end
end

sub2XTest(1:N-1,:) = [];

% sub3
sub3XTest = ones(NumWins(xLen,fs,winLen,winDisp),v2*N*f+1);

for j = 1:v2
    %disp(j);
    for i = N:NumWins(xLen,fs,winLen,winDisp)
        sub3XTest(i,((j-1)*N*f+2):(j*N*f)+1) = [sub3_1_60Testamp{j}(i-N+1:i) sub3_60_100Testamp{j}(i-N+1:i) ...
            sub3_100_200Testamp{j}(i-N+1:i)]; %insert data into R
    end
end

sub3XTest(1:N-1,:) = [];