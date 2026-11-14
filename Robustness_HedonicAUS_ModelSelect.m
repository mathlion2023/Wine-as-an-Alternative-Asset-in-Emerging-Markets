%% Robustness_HedonicAUS_ModelSelect.m
% -------------------------------------------------------------------------
% Referee 2 item 5 robustness run: mirrors fitIndices_autoModels_v2.m +
% rebuild_residualstd.m exactly, but on MasterData_USD_2019_2024_4country_HedonicAUS.mat
% (comment corrected 2026-10-10, code-review finding #15 -- line 12 below
% already loaded the correct current filename; only this comment was stale)
% (AUS_wine_USD replaced by the Langton's chained-hedonic-index series --
% see Build_HedonicAUS_MonthlyIndex.m) instead of the original master
% dataset. All outputs are suffixed _HedonicAUS so nothing here overwrites
% the manuscript's actual Table 4/5/6/ResidualSTD.mat.
% -------------------------------------------------------------------------
clc; clear all;

load('MasterData_USD_2019_2024_4country_HedonicAUS.mat', 'M');
srcCols    = M.Properties.VariableNames(2:end);
indexNames = string(srcCols)';
data = M{:, srcCols};

R = diff(log(data)); % T x N returns

OUT = fit_auto_vol_models(R, indexNames);

write_latex_model_table_all(OUT, indexNames, 'tab_selected_models_all_HedonicAUS.tex');
write_latex_model_table(OUT, indexNames, 'tab_selected_models_HedonicAUS.tex');
write_full_model_comparison(OUT, indexNames, 'tab_allmodels_18_HedonicAUS.tex', 'tab_allmodels_18_HedonicAUS.csv');
write_figarch_only_table(OUT, indexNames, 'tab_figarch_all_18_HedonicAUS.tex', 'tab_figarch_all_18_HedonicAUS.csv');

for i = 1:numel(indexNames)
    fprintf('%-16s -> %s\n', indexNames(i), OUT.choice{i});
end

stdResid = OUT.stdz;
names = indexNames;
save('ResidualSTD_HedonicAUS.mat', 'stdResid', 'names');
fprintf('Saved ResidualSTD_HedonicAUS.mat and tab_*_HedonicAUS.tex/.csv\n');

% Headline comparison line for AUS_wine_USD specifically
iAUS = find(indexNames == "AUS_wine_USD", 1);
if ~isempty(iAUS)
    fprintf('\nAUS_wine_USD (hedonic-swapped) selected model: %s\n', OUT.choice{iAUS});
    disp(OUT.params{iAUS});
end
