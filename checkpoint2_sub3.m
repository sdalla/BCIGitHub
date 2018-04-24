%% plot all channels in time domain to make sure nothing crazy is going on
for i = 1:64
    figure(1)
    subplot(8,8,i)
    plot(Sub3_Training_ecog{i})
    ylim([-2e4 2e4])
    set(gca,'YTickLabel',[]);
    set(gca,'XTickLabel',[]);
end

%% Identifying channels with Excessive Line Noise
%sampling freq
Fs = 1000;
%length of the sample
l = length(Sub3_Training_ecog{1});
%freq array of sample
f = Fs*(0:(l/2))/l;
ind = 1;
for i = 1:64
  
fft_sub3 = fft(Sub3_Training_ecog{i});
fft_mag = abs((fft_sub3));
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
data = Sub3_Training_ecog;

RMSV = cellfun(@rms,data);
logRMSV = log10(RMSV);

% Normalize  by calculating z-scores
zscores = zscore(logRMSV);
scatter(1:length(zscores),zscores);

% Calculate p-values using a Gaussian distribution fitted to the data
[h,p] = ttest(logRMSV);
channelsExcl = find(logRMSV > (mean(logRMSV)+2*std(logRMSV)));
% this returns channels 23, 38, 58

%%
