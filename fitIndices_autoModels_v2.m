
clc; clear all;
load('projectdata.mat');
load('projectdate.mat'); % T×1 datetime

% Data=table2array(data);
R   = diff(log(data)); % T×N returns
% R is T x N (e.g., monthly log returns of the 9 indices)
indexNames = ["SAfw","Ausfw","Argfw","chifw","SP500","MSCIEM","Bond","Gold","Liv-ex100"]';
%%
OUT = fit_auto_vol_models(R, indexNames);
%%

write_latex_model_table_all(OUT, indexNames, 'tab_selected_models_all.tex');

%%
write_latex_model_table(OUT, indexNames, 'tab_selected_models.tex');

% Example: inspect AUSFW choice and parameters
i = find(indexNames=="Ausfw");
OUT.choice{i}
OUT.params{i}    % table of parameters & s.e. (FIGARCH includes d and its se)
OUT.stats{i}

% Reuse: one-step-ahead variance forecast (naïve) from chosen model
sigma_t1 = sqrt(OUT.sigma2(end,:));   % conditional stdev at last t
