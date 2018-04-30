
%%
load('Sub1_Leaderboard_ecog.mat')
load('Sub2_Leaderboard_ecog.mat')
load('Sub3_Leaderboard_ecog.mat')

%%
Sub1cellFull = cell2mat(Sub1_Leaderboard_ecog);
Sub2cellFull = cell2mat(Sub2_Leaderboard_ecog);
Sub3cellFull = cell2mat(Sub3_Leaderboard_ecog);

Sub1cellHalf = Sub1cellFull(1:end/2,:);
Sub2cellHalf = Sub2cellFull(1:end/2,:);
Sub3cellHalf = Sub3cellFull(1:end/2,:);

halfTest_ecog{1} = Sub1cellHalf;
halfTest_ecog{2} = Sub2cellHalf;
halfTest_ecog{3} = Sub3cellHalf;
