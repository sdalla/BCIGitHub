function excludedChannels = ExcLineNoise(data,outliers,v)
    %sampling freq
    Fs = 1000;
    %length of the sample
    l = length(data{1});
    %freq array of sample
    f = Fs*(0:(l/2))/l;
    ind = 1;
    for i = 1:v
        if i == outliers
            continue
        end
        
        fft_sub1 = fft(data{i});
        fft_mag = abs((fft_sub1));
        fft_mag_plot = fft_mag(1:l/2+1); 

        noise(ind) = mean(fft_mag(find(f==60)-75:find(f==60)+75));
        ind = ind + 1;
    end
    figure()
    plot(noise/median(noise),'o')

    %channel numbers that have line noises that have statistically significant
    %deviations
    excludedChannels = find(noise/median(noise) > mean(noise/median(noise)) + 2*std(noise/median(noise)));
end