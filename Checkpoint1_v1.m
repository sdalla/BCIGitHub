
%% Feature Extration Code (taken from HW3)
% gets the number of windows given the length and displacement
% winLen and winDisp are in time, fs is sampling freq, xLen is length in
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
%%