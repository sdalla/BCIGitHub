load('predicted_dg041.mat')
for i = 1:3
    for j = 1:5
        output{i}(:,j) = medfilt1(predicted_dg{i}(:,j),1000);
    end
end

figure()
plot(predicted_dg{1}(:,1))
hold on
plot(output{1}(:,1))
legend('No filter','Filter')
predicted_dg = output;
save predicted_dg_filt predicted_dg