function out = MovingWinFeats(x, fs, winLen, winDisp, featFn)
wins = length(0:winDisp*fs:length(x))-(winLen/winDisp);

    for i = 1:wins
        data = x(1+(i-1)*winDisp*fs:(fs*winDisp*(i-1)+winLen*fs));
        out(i) = featFn(data);
    end
end