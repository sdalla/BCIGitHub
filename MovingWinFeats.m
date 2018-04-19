function out = MovingWinFeats(x, fs, winLen, winDisp, featFn)
wins = length(0:winDisp*fs:length(x))-(winLen/winDisp);

    for i = 1:wins
        data = x(round(1+(i-1)*winDisp*fs):round((fs*winDisp*(i-1)+winLen*fs)));
        out(i) = featFn(data);
    end
end
