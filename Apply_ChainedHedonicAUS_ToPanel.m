%% Apply_ChainedHedonicAUS_ToPanel.m (2026-10-10, peer-review item 4)
% -------------------------------------------------------------------------
% Takes Build_ChainedHedonicAUS_MonthlyIndex.m's AUD, 2004-2025 chained
% index and substitutes it for AUS_wine_USD in the SHORT/baseline panel,
% mirroring Build_HedonicAUS_MonthlyIndex.m's existing pooled-time-dummy
% version so the two robustness checks are directly comparable (same
% window, same currency conversion, same substitution target).
% -------------------------------------------------------------------------

clear; clc;
WINDOW_START = datetime(2019,6,1);
WINDOW_END   = datetime(2024,11,1);

load('ChainedHedonicAUS.mat', 'chainedAUS');
win = chainedAUS(chainedAUS.Period >= WINDOW_START & chainedAUS.Period <= WINDOW_END, :);
assert(win.Period(1) == WINDOW_START, 'Window does not start at the expected panel start date.');
levelAUD = exp(win.LogIndex_Chained - win.LogIndex_Chained(1)); % rebase log-index to 0 at window start first

optsBB = detectImportOptions('Wine DATA.xlsx', 'Sheet', 'Sheet1');
optsBB.VariableNamesRange = 'A6'; optsBB.DataRange = 'A7';
Tbb = readtable('Wine DATA.xlsx', optsBB);
Tbb.Properties.VariableNames{1} = 'Date';
Tbb.Period = dateshift(datetime(Tbb.Date), 'start', 'month');
[tfx, locx] = ismember(win.Period, Tbb.Period);
assert(all(tfx), 'AUDUSD FX series does not fully cover the window.');
fx = Tbb.AUDUSD(locx(tfx));
levelUSD = levelAUD .* fx;
levelUSD_100 = 100 * levelUSD / levelUSD(1);

Orig = load('MasterData_USD_2019_2024_4country.mat', 'M');
M = Orig.M;
assert(isequal(M.Period, win.Period), 'Date alignment mismatch between M and the chained hedonic AUS series.');
origAUS = M.AUS_wine_USD;
M.AUS_wine_USD = levelUSD_100;
save('MasterData_USD_2019_2024_4country_ChainedHedonicAUS.mat', 'M');

retOrig = diff(log(origAUS));
retChained = diff(log(levelUSD_100));
cc = corr(retOrig, retChained);
fprintf('\nOriginal (hybrid) vs chained-hedonic AUS monthly log-return correlation: %.3f\n', cc);
fprintf('  Original AUS_wine_USD:        mean %.4f, std %.4f\n', mean(retOrig), std(retOrig));
fprintf('  Chained-hedonic AUS_wine_USD: mean %.4f, std %.4f\n', mean(retChained), std(retChained));

cmp = table(M.Period(2:end), retOrig, retChained, 'VariableNames', ...
    {'Period','AUS_wine_USD_original_ret','AUS_wine_USD_chained_ret'});
writetable(cmp, 'tab_chainedhedonicAUS_vs_original_AUS.csv');
fprintf('Saved MasterData_USD_2019_2024_4country_ChainedHedonicAUS.mat, tab_chainedhedonicAUS_vs_original_AUS.csv\n');
