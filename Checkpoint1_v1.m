%% Feature Extration Code (taken from HW3)
% gets the number of windows given the length and displacement
% winLen and winDisp are in time, fs is sampling freq, xLen is length in
% samples
NumWins = @(xLen, fs, winLen, winDisp) length(0:winDisp*fs:xLen)-(winLen/winDisp);

% Line length fxn
LLFn = @(x) sum(abs(diff(x)));
