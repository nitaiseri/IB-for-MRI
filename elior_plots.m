
addpath(genpath('/ems/elsc-labs/mezer-a/code/elior'));

%% Visualize result spatial functions
% Display group-averaged gradients (and possibly individual subjects
% gradients) of 1 or more ROIs, in 1 or more subject-groups

group_ids = 1:2; % subject groups (rows of RG)
roi_ids = 1; % ROIs (columns of RG)
rg = RG(group_ids,roi_ids);
PC = 1:3;
ind = [0 0 0 0]; % if more than one subject, indicate for each group/roi whether to display individual gradients

y_lim = [];%[.71 .88];
fig1 = showRG(rg,'errorType',2,'ind',ind,'PC',PC,...
    'markershapes',1,'legend',1,'labels',1,'ylim',y_lim);

fig1.WindowState = 'maximized';