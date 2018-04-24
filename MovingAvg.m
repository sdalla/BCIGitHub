%% Function takes moving average of each finger and returns filtered output
% Input should be AxB matrix where A is number of time points and B is
% number of fingers
function filtSubData = MovingAvg(subData)
    fs = 1000; %sampling rate of data
    coeffMAvg = ones(1, fs)/fs;
    fDelay = (length(coeffMAvg)-1)/2; %delay caused by using Moving Avg Filter
    %filtSubData = zeros(size(subData,1),size(subData,2));
    subDataPad = vertcat(ones(500,1), subData);
    for i = 1 : size(subDataPad,2);
        %apply filter to each finger and save result
        filtSubData(:,i) = filter(coeffMAvg, 1, subDataPad(:,i));
    end
    figure()
    plot(subData(:,1))
    hold on
    plot(filtSubData(:,1))
    legend('Raw Data','Filtered Data')
    
end