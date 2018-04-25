load('predicted_dg041.mat')
for i = 1:3
    for j = 1:5
        output{i}(:,j) = medfilt1(predicted_dg{i}(:,j),1000);
    end
end
%%
figure(1)
plot(predicted_dg{1}(:,1))
hold on
plot(output{1}(:,1))
legend('No filter','Filter')

figure(2)
plot(predicted_dg{1}(:,2))
hold on
plot(output{1}(:,2))
legend('No filter','Filter')

figure(3)
plot(predicted_dg{1}(:,3))
hold on
plot(output{1}(:,3))
legend('No filter','Filter')

figure(5)
plot(predicted_dg{1}(:,5))
hold on
plot(output{1}(:,5))
legend('No filter','Filter')
%predicted_dg = output;
%save predicted_dg_filt predicted_dg