%% Rebuild ResidualSTD.mat from the corrected per-series model selection
% (guardrail fix in fit_auto_vol_models.m: FIGARCH is excluded, not merely
% demoted, when |t(d)| < 1.96). Replaces the previous ResidualSTD.mat, which
% had been built by force-fitting FIGARCH to every series regardless of
% selection (see DCC_Analysis.m 's sequential-overwrite pattern).
clc; clear all;
% Repointed (2026-08-25) at the USD master dataset (Build_MasterData_USD.m
% output), Jan 2013-Nov 2024, replacing projectdata.mat/projectdate.mat.
% Runs on ALL 18 series in the master dataset (2026-08-25, per user
% instruction), not just the original manuscript's 9-asset subset.
load('MasterData_USD_2013_2024.mat', 'M');
srcCols    = M.Properties.VariableNames(2:end);
indexNames = string(srcCols)';
data = M{:, srcCols};

R = diff(log(data));

OUT = fit_auto_vol_models(R, indexNames);

stdResid = OUT.stdz;
names = indexNames;

for i = 1:numel(indexNames)
    fprintf('%-10s -> %s\n', indexNames(i), OUT.choice{i});
end

save('ResidualSTD.mat', 'stdResid', 'names');
fprintf('Saved corrected ResidualSTD.mat\n');
