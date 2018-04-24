function excludedChannels = AbnormAmpDist(data,outliers,v)
    data(outliers) = [];
    RMSV = cellfun(@rms,data);
    logRMSV = log10(RMSV);

    % Normalize  by calculating z-scores
    zscores = zscore(logRMSV);
    scatter(1:(v-length(outliers)),zscores);

    % Calculate p-values using a Gaussian distribution fitted to the data
    %[h,p] = ttest(logRMSV);
    excludedChannels = find(logRMSV > (mean(logRMSV)+2*std(logRMSV)));
end