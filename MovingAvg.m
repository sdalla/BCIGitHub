%% Function takes moving average of each finger and returns filtered output
% Input should be AxB matrix where A is number of time points and B is
% number of fingers
function filtSubData = MovingAvg(subData)
    fs = 1000; %sampling rate of data
    coeffMAvg = ones(1, fs)/fs;
    fDelay = (length(coeffMAvg)-1)/2; %delay caused by using Moving Avg Filter
    filtSubData = zeros(size(subData,1),size(subData,2));
    for i = 1 : size(subData,2)
        %apply filter to each finger and save result
        filtSubData(:,i) = filter(coeffMAvg, 1, subData(:,i));
    end
end