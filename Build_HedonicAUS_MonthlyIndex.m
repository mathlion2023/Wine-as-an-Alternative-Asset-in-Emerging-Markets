%% Build_HedonicAUS_MonthlyIndex.m
% -------------------------------------------------------------------------
% Referee 2, item 5: chained hedonic index as a robustness check against
% structural breaks (Triplett, 2004), "does it change conclusions?"
%
% Wine Data\Step1a_PrepareWine.m already built a genuine chained hedonic
% price index for Australia from raw Langton's lot-level transactions
% (event fixed effects + wine-name FE + vintage FE + lot controls,
% R^2=0.898, N=445,137) -- saved as mkt_index inside
% "Wine Data\Output\wine_panel.mat". This script turns that event-level
% index into a monthly, USD-denominated series directly comparable to
% AUS_wine_USD in MasterData_USD_2013_2024.mat, and swaps it in to make an
% alternate master dataset for the robustness run.
%
% CURRENCY: Langton's is an Australian auction house, so AUD is not an
% assumption here (unlike the original AUS_wine_USD, whose AUD label is
% flagged elsewhere as unconfirmed) -- the hedonic index inherits AUD
% directly from the raw lot prices it was estimated on.
%
% MONTHLY AGGREGATION (a methodological choice, stated explicitly):
% within each calendar month, take the LAST auction event's hedonic index
% level (a month-end snapshot); months with no Langton's auction that
% month carry the previous month's level forward. This mirrors how the
% original EmergingCountryIndex.xlsx country series are themselves
% interpolated onto a monthly grid from irregular auction dates (see
% CLAUDE.md), so the two AUS series are aggregated the same way.
%
% Output (CORRECTED 2026-10-10, code-review finding #15 -- this docblock
% still named the pre-2026-10-08 LONG-panel filename; the script was
% re-pointed to the SHORT/4-country panel per the "UPDATED 2026-10-08"
% comment below but this Output section was never updated to match. The
% stale MasterData_USD_2013_2024_HedonicAUS.mat this used to point to has
% been renamed MasterData_USD_2013_2024_HedonicAUS_STALE_PRE_SHORTPANEL_REPOINT.mat
% so it is not mistaken for current output):
%   MasterData_USD_2019_2024_4country_HedonicAUS.mat  (variable M, same
%                                               column shape as
%                                               MasterData_USD_2019_2024_4country.mat,
%                                               with AUS_wine_USD replaced)
%   MasterData_USD_LogReturns_HedonicAUS.csv
%   tab_hedonicAUS_vs_original_AUS.csv        (side-by-side return series
%                                               + correlation, for the
%                                               "does it change anything
%                                               even at the raw-data level"
%                                               sanity check)
% -------------------------------------------------------------------------

clear; clc;

% UPDATED 2026-10-08: re-pointed at the SHORT/baseline panel (all four
% wine countries, Jun2019-Nov2024), now primary (Section 5.2), instead of
% the LONG panel (which no longer carries a Jan-2013 start -- Chile's
% hybrid index has no native data at Jan-2013, so the corrected LONG panel
% starts Feb-2013; see Section 3.1/Build_MasterData_USD_v2.m).
WINDOW_START = datetime(2019,6,1);
WINDOW_END   = datetime(2024,11,1);

%% ---- 1. Load the chained hedonic index (event-level, 2004-2025, AUD) ----
S = load(fullfile('Wine Data','wine_panel.mat'), 'mkt_index');
mkt_index = sortrows(S.mkt_index, 'Date');
fprintf('Loaded hedonic index: %d event dates, %s to %s\n', height(mkt_index), ...
    datestr(min(mkt_index.Date)), datestr(max(mkt_index.Date)));

%% ---- 2. Monthly month-end snapshot, forward-filled over the FULL range ----
% (forward-filled over the full 2004-2025 range first, then sliced to the
% 2013-2024 window below, so the window's first month isn't left as a gap
% just because it happens to fall before that month's first auction date)
mkt_index.YM = dateshift(mkt_index.Date, 'start', 'month');
[ym, ~, g] = unique(mkt_index.YM);
lastLogIdx = splitapply(@(x) x(end), mkt_index.CumMktReturn, g);

allMonths = (dateshift(min(mkt_index.Date),'start','month') : calmonths(1) : ...
             dateshift(max(mkt_index.Date),'start','month'))';
[tf, loc] = ismember(allMonths, ym);
levelAUD = NaN(numel(allMonths), 1);
levelAUD(tf) = exp(lastLogIdx(loc(tf)));
nFilled = 0;
for i = 2:numel(levelAUD)
    if isnan(levelAUD(i)), levelAUD(i) = levelAUD(i-1); nFilled = nFilled + 1; end
end
assert(~isnan(levelAUD(1)), 'First month of the full range has no level to seed forward-fill.');
fprintf('Monthly series built: %d months (%d auction months, %d forward-filled)\n', ...
    numel(allMonths), sum(tf), nFilled);

full = table(allMonths, levelAUD, 'VariableNames', {'Period','LevelAUD'});
win = full(full.Period >= WINDOW_START & full.Period <= WINDOW_END, :);
assert(~any(isnan(win.LevelAUD)), 'Gaps remain in the Jan2013-Nov2024 window after forward-fill.');
fprintf('Window slice: %d months (%s to %s)\n', height(win), datestr(min(win.Period)), datestr(max(win.Period)));

%% ---- 3. AUD -> USD, same method/source as Build_MasterData_USD.m ----
optsBB = detectImportOptions('Wine DATA.xlsx', 'Sheet', 'Sheet1');
optsBB.VariableNamesRange = 'A6';
optsBB.DataRange          = 'A7';
Tbb = readtable('Wine DATA.xlsx', optsBB);
Tbb.Properties.VariableNames{1} = 'Date';
Tbb.Date = datetime(Tbb.Date);
Tbb.Period = dateshift(Tbb.Date, 'start', 'month');

[tfx, locx] = ismember(win.Period, Tbb.Period);
assert(all(tfx), 'AUDUSD FX series does not fully cover the Jan2013-Nov2024 window.');
fx = Tbb.AUDUSD(locx(tfx));
win.LevelUSD = win.LevelAUD .* fx;   % AUDUSD quoted USD-per-AUD (same convention as Build_MasterData_USD.m)

%% ---- 4. Rebase to 100 @ panel start (same convention as the rest of M) ----
assert(win.Period(1) == WINDOW_START, 'Window does not start at the expected panel start date.');
win.LevelUSD_100 = 100 * win.LevelUSD / win.LevelUSD(1);

%% ---- 5. Swap into a copy of the master dataset ----
Orig = load('MasterData_USD_2019_2024_4country.mat', 'M');
M = Orig.M;
assert(isequal(M.Period, win.Period), 'Date alignment mismatch between M and the hedonic AUS series.');
origAUS = M.AUS_wine_USD;
M.AUS_wine_USD = win.LevelUSD_100;

save('MasterData_USD_2019_2024_4country_HedonicAUS.mat', 'M');

lvl = table2array(M(:,2:end));
ret = diff(log(lvl));
R = array2table(ret, 'VariableNames', M.Properties.VariableNames(2:end));
R = addvars(R, M.Period(2:end), 'Before', 1, 'NewVariableNames', 'Period');
writetable(R, 'MasterData_USD_LogReturns_HedonicAUS.csv');

%% ---- 6. Sanity-check comparison: original vs hedonic AUS return series ----
retOrig = diff(log(origAUS));
retHed  = diff(log(win.LevelUSD_100));
cc = corr(retOrig, retHed);
fprintf('\nOriginal vs hedonic AUS monthly log-return correlation: %.3f\n', cc);
fprintf('  Original AUS_wine_USD: mean %.4f, std %.4f\n', mean(retOrig), std(retOrig));
fprintf('  Hedonic  AUS_wine_USD: mean %.4f, std %.4f\n', mean(retHed),  std(retHed));

cmp = table(M.Period(2:end), retOrig, retHed, 'VariableNames', ...
    {'Period','AUS_wine_USD_original_ret','AUS_wine_USD_hedonic_ret'});
writetable(cmp, 'tab_hedonicAUS_vs_original_AUS.csv');
fprintf('Saved MasterData_USD_2019_2024_4country_HedonicAUS.mat, MasterData_USD_LogReturns_HedonicAUS.csv, tab_hedonicAUS_vs_original_AUS.csv\n');
