
clc; clear all;
% Repointed (2026-08-25) at the USD master dataset (Build_MasterData_USD.m
% output), Jan 2013-Nov 2024, replacing projectdata.mat/projectdate.mat
% (Jun 2019-Nov 2024, 66 months). This script (not Main_model_check.m)
% is the one that writes tab_selected_models_4country.tex/_all.tex, i.e. the
% manuscript's actual Table 4/5/6 source -- and it already uses the
% guardrail-fixed fit_auto_vol_models.m. Runs on ALL 18 series in the
% master dataset (2026-08-25, per user instruction), not just the
% original manuscript's 9-asset subset.
load('MasterData_USD_2019_2024_4country.mat', 'M');
srcCols    = M.Properties.VariableNames(2:end);
indexNames = string(srcCols)';
data = M{:, srcCols};

% Data=table2array(data);
R   = diff(log(data)); % T×N returns
% R is T x N (e.g., monthly log returns of the 18 indices)
%%
OUT = fit_auto_vol_models(R, indexNames);
%%

write_latex_model_table_all(OUT, indexNames, 'tab_selected_models_all_4country.tex');

%%
write_latex_model_table(OUT, indexNames, 'tab_selected_models_4country.tex');

%% Table 4 (all four candidates per series, not just the winner) and
% Table 5 (FIGARCH-only fit for every series) -- 2026-09-03, resolves the
% "PENDING REGENERATION" placeholders in QF_Paper_Revised.tex. OUT.all{i}
% already holds every trial fit_auto_vol_models.m estimated; no prior
% script persisted it to disk.
write_full_model_comparison(OUT, indexNames, 'tab_allmodels_4country.tex', 'tab_allmodels_4country.csv');
write_figarch_only_table(OUT, indexNames, 'tab_figarch_all_4country.tex', 'tab_figarch_all_4country.csv');

% Example: inspect AUSFW choice and parameters
i = find(indexNames=="Ausfw");
OUT.choice{i}
OUT.params{i}    % table of parameters & s.e. (FIGARCH includes d and its se)
OUT.stats{i}

% Reuse: one-step-ahead variance forecast (naïve) from chosen model
sigma_t1 = sqrt(OUT.sigma2(end,:));   % conditional stdev at last t
