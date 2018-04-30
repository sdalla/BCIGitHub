load('Sub1_Training_dg.mat')
load('Sub2_Training_dg.mat')
load('Sub3_Training_dg.mat')


for dg = 1:5
    training_dg{dg,1} = Sub1_Training_dg{dg};
end
for dg = 1:5
    training_dg{dg,2} = Sub2_Training_dg{dg};
end
for dg = 1:5
    training_dg{dg,3} = Sub3_Training_dg{dg};
end


for subj = 1:3
    for i = 1:5
        [idx1,c1] = kmeans(training_dg{i,subj},2);
        [sortc, idxc1] = sort(c1);
        idx{i,subj} = idx1;
        idxc{i,subj} = idxc1(1);
    end
end

save idx idx
save idxc idxc
save training_dg training_dg