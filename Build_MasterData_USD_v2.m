%% Build_MasterData_USD_v2.m (2026-10-08)
% -------------------------------------------------------------------------
% Replaces Build_MasterData_USD.m as the source of the wine-index master
% dataset, fixing two data-integrity problems discovered this session:
%
% 1. SOUTH AFRICA HAS NO DATA BEFORE JUNE 2019. Strauss & Co. (SA's only
%    data source) was founded in 2019; EmergingCountryIndex.xlsx's "SA"
%    column nonetheless carried values back to Jan 2013 from an unknown,
%    unconfirmed source -- and even its post-2019 segment did not match
%    the original manuscript's genuine Strauss-based SA series (checked
%    directly: rebasing both to 100 at Jun-2019, EmergingCountryIndex's
%    SA swings to 72/104/80/88 within 6 months while the original
%    Projectdata/WineData2025 RS-sheet SA stays near 92-107 -- not the
%    same series, not even the same methodology). EmergingCountryIndex.xlsx
%    is no longer used as a wine-index source at all (see point 2).
%
% 2. CURRENCY MIXING IN THE RAW LOT DATA. "Data Australia.xlsx" (the
%    Verdickt-supplied AUS/ARG/CHI lot file) pools many auction houses.
%    Only Langton's (Australia's domestic house) is AUD; Argentina and
%    Chile have no dominant domestic house in this dataset at all --
%    essentially every ARG/CHI row, and ~10% of AUS rows, are
%    international houses (Acker Merrall & Condit, Zachys, K&L Wines,
%    Sotheby's, Christie's, Heritage Auctions, Spectrum, ...) that sell in
%    USD. The original Build_MasterData_USD.m divided the whole AUS/ARG/CHI
%    index by AUDUSD/USDARS/USDCLP regardless, which would have run
%    Argentina's peso-devaluation FX history through a wine index that a
%    USD-paying buyer never actually experienced. Fixed at the lot level
%    in Build_RS_Hybrid_AllCountries.m (run that script first) -- its
%    AUSFW_HY/ARGFW_HY/CHIFW_HY outputs are already USD; only SAFW_HY is
%    still local currency (ZAR) and needs FX conversion here.
%
% Consequently this script sources wine indices from
% RS_Hybrid_Comparison_18.mat (the self-built Fogarty & Jones 2011 hybrid
% hedonic-repeat-sales index, built directly from raw lot data) instead
% of EmergingCountryIndex.xlsx, and produces TWO panels instead of one:
%
%   MasterData_USD_2013_2024.xlsx/.mat       - LONG panel, 2013-01 to
%       2024-11 (143 months). AUS/ARG/CHI wine + all benchmarks. SA wine
%       EXCLUDED (no real data exists for 72 of these 143 months).
%       Kept under the original filename so every script that already
%       reads M.Properties.VariableNames(2:end) dynamically (
%       Main_model_check.m, fitIndices_autoModels_v2.m, DCC_Analysis.m,
%       Spillover_Analysis.m) just runs on 17 series with no further
%       change needed.
%
%   MasterData_USD_2019_2024_4country.xlsx/.mat - SHORT/baseline panel,
%       2019-06 to 2024-11 (66 months, SA's real native window). All FOUR
%       wine countries + all benchmarks. This is the paper's core
%       4-country comparison, on the window the data can actually
%       support -- essentially the original submission's design,
%       re-run on the bug-fixed pipeline and the corrected index.
%
% Run Build_RS_Hybrid_AllCountries.m BEFORE this script.
% -------------------------------------------------------------------------

clear; clc;

%% ---- 1. Wine indices from the self-built hybrid index (not EmergingCountryIndex.xlsx) ----
load('RS_Hybrid_Comparison_18.mat', 'T');
Twine = T(:, {'Period','SAFW_HY','AUSFW_HY','ARGFW_HY','CHIFW_HY'});
Twine.Properties.VariableNames = {'Period','SA','AUS','ARG','CHL'};
Twine.Period = dateshift(datetime(Twine.Period), 'start', 'month');

%% ---- 2. Bloomberg benchmark + FX panel ----
optsBB = detectImportOptions('Wine DATA.xlsx', 'Sheet', 'Sheet1');
optsBB.VariableNamesRange = 'A6';
optsBB.DataRange          = 'A7';
Tbb = readtable('Wine DATA.xlsx', optsBB);
Tbb.Properties.VariableNames{1} = 'Date';
Tbb.Date = datetime(Tbb.Date);
Tbb.Period = dateshift(Tbb.Date, 'start', 'month');
Tbb = Tbb(Tbb.Period >= min(Twine.Period) & Tbb.Period <= max(Twine.Period), :);

%% ---- 3. Merge on Period ----
T = innerjoin(Twine, Tbb, 'Keys', 'Period');
assert(height(T) == height(Twine), 'Merge dropped rows -- Wine DATA.xlsx does not fully cover the wine-index window.');

%% ---- 4. Convert to USD ----
% AUS/ARG/CHL wine are ALREADY USD (lot-level fix in
% Build_RS_Hybrid_AllCountries.m) -- no FX conversion. SA wine is still
% ZAR (Strauss & Co.'s own pricing) and needs the usual conversion.
usd  = @(localLevel, fx) localLevel ./ fx;   % local-per-USD quote (ZAR/ARS/CLP)
usdM = @(localLevel, fx) localLevel .* fx;   % USD-per-local quote (AUD/EUR/GBP)

M = table();
M.Period = T.Period;

M.SA_wine_USD  = usd(T.SA, T.USDZAR);
M.AUS_wine_USD = T.AUS;   % already USD
M.ARG_wine_USD = T.ARG;   % already USD
M.CHL_wine_USD = T.CHL;   % already USD

M.LivEx100_USD = usdM(T.LIVEX100, T.GBPUSD);

M.JALSH_USD  = usd(T.JALSH_TR,  T.USDZAR);
M.ASX200_USD = usdM(T.ASX200_TR, T.AUDUSD);
M.MERVAL_USD = usd(T.MERVAL_TR, T.USDARS);
M.IPSA_USD   = usd(T.IPSA_TR,   T.USDCLP);

M.CAC_USD  = usdM(T.CAC_TR,  T.EURUSD);
M.MIB_USD  = usdM(T.MIB_TR,  T.EURUSD);
M.IBEX_USD = usdM(T.IBEX_TR, T.EURUSD);
if all(isnan(T.PSI_TR))
    warning('PSI_TR is empty in this Wine DATA.xlsx download; falling back to PSI_PX.');
    M.PSI_USD = usdM(T.PSI_PX, T.EURUSD);
else
    M.PSI_USD = usdM(T.PSI_TR, T.EURUSD);
end
M.DAX_USD = usdM(T.DAX, T.EURUSD);

M.SPX_USD       = T.SPTR_TR;
M.MSCIEM_USD    = T.MXEF_TR;
M.GlobalAgg_USD = T.GLOBAGG_TR;
M.Gold_USD      = T.GOLD;

%% ---- 5a. LONG panel: 2013-2024, drop SA (no real data for 72/143 months) ----
Mlong = removevars(M, 'SA_wine_USD');
seriesNames0 = Mlong.Properties.VariableNames(2:end);
nValid = sum(~isnan(Mlong{:, seriesNames0}), 1);
% Chile's hybrid index has no native data at Jan-2013 (its first
% transaction is Feb-2013) -- a 1-month leading gap, not a short series.
% Trim the panel's START to the latest "first valid month" across all
% kept series (rather than dropping the series outright, which the old
% <full-coverage> filter would have done to Chile for the sake of a
% single edge-of-sample month), then re-check full coverage from there.
firstValidRow = nan(1, numel(seriesNames0));
for j = 1:numel(seriesNames0)
    firstValidRow(j) = find(~isnan(Mlong.(seriesNames0{j})), 1, 'first');
end
commonStartRow = max(firstValidRow);
if commonStartRow > 1
    warning('LONG panel: trimming %d leading month(s) so every series has a common start (first full-coverage month: %s).', ...
        commonStartRow - 1, datestr(Mlong.Period(commonStartRow), 'yyyy-mm'));
    Mlong = Mlong(commonStartRow:end, :);
end
nValid2 = sum(~isnan(Mlong{:, seriesNames0}), 1);
short = seriesNames0(nValid2 < height(Mlong));
if ~isempty(short)
    warning('LONG panel: dropping series still short of full coverage after trimming: %s', strjoin(short, ', '));
    Mlong = removevars(Mlong, short);
end
seriesNames = Mlong.Properties.VariableNames(2:end);
base = Mlong{1, seriesNames};
assert(~any(isnan(base)), 'LONG panel: series NaN at panel start (%s), cannot rebase: %s', ...
    datestr(Mlong.Period(1),'yyyy-mm'), strjoin(seriesNames(isnan(base)), ', '));
Mlong{:, seriesNames} = 100 * Mlong{:, seriesNames} ./ base;

% Keep variable name 'M' in the saved .mat, since every downstream script
% (Main_model_check.m, fitIndices_autoModels_v2.m, DCC_Analysis.m,
% Spillover_Analysis.m, portfolio scripts) loads 'M' by that name.
writetable(Mlong, 'MasterData_USD_2013_2024.xlsx');
M = Mlong;
save('MasterData_USD_2013_2024.mat', 'M');

lvl = table2array(Mlong(:,2:end));
ret = diff(log(lvl));
R = array2table(ret, 'VariableNames', Mlong.Properties.VariableNames(2:end));
R = addvars(R, Mlong.Period(2:end), 'Before', 1, 'NewVariableNames', 'Period');
writetable(R, 'MasterData_USD_LogReturns.csv');

fprintf('\nLONG panel (SA excluded): %d months (%s to %s), %d series.\n', ...
    height(Mlong), datestr(min(Mlong.Period),'yyyy-mm'), datestr(max(Mlong.Period),'yyyy-mm'), width(Mlong)-1);
disp(Mlong(1:3,:));

%% ---- 5b. SHORT/baseline panel: 2019-06 to 2024-11, all 4 wine countries ----
shortStart = datetime(2019,6,1);
% Rebuild the full (unrebased) 18-series table for the short window from
% T directly, since Mlong above already dropped SA and rebased to the
% LONG panel's own base.
Tshort = T(T.Period >= shortStart, :);
Ms = table();
Ms.Period = Tshort.Period;
Ms.SA_wine_USD  = usd(Tshort.SA, Tshort.USDZAR);
Ms.AUS_wine_USD = Tshort.AUS;
Ms.ARG_wine_USD = Tshort.ARG;
Ms.CHL_wine_USD = Tshort.CHL;
Ms.LivEx100_USD = usdM(Tshort.LIVEX100, Tshort.GBPUSD);
Ms.JALSH_USD  = usd(Tshort.JALSH_TR,  Tshort.USDZAR);
Ms.ASX200_USD = usdM(Tshort.ASX200_TR, Tshort.AUDUSD);
Ms.MERVAL_USD = usd(Tshort.MERVAL_TR, Tshort.USDARS);
Ms.IPSA_USD   = usd(Tshort.IPSA_TR,   Tshort.USDCLP);
Ms.CAC_USD  = usdM(Tshort.CAC_TR,  Tshort.EURUSD);
Ms.MIB_USD  = usdM(Tshort.MIB_TR,  Tshort.EURUSD);
Ms.IBEX_USD = usdM(Tshort.IBEX_TR, Tshort.EURUSD);
if all(isnan(Tshort.PSI_TR))
    Ms.PSI_USD = usdM(Tshort.PSI_PX, Tshort.EURUSD);
else
    Ms.PSI_USD = usdM(Tshort.PSI_TR, Tshort.EURUSD);
end
Ms.DAX_USD = usdM(Tshort.DAX, Tshort.EURUSD);
Ms.SPX_USD       = Tshort.SPTR_TR;
Ms.MSCIEM_USD    = Tshort.MXEF_TR;
Ms.GlobalAgg_USD = Tshort.GLOBAGG_TR;
Ms.Gold_USD      = Tshort.GOLD;

seriesNames0s = Ms.Properties.VariableNames(2:end);
nValidS = sum(~isnan(Ms{:, seriesNames0s}), 1);
shortDrop = seriesNames0s(nValidS < height(Ms));
if ~isempty(shortDrop)
    warning('SHORT panel: dropping series with <%d months of data: %s', height(Ms), strjoin(shortDrop, ', '));
    Ms = removevars(Ms, shortDrop);
end
assert(all(Ms.Period(1) == shortStart), 'SHORT panel does not start Jun 2019.');
seriesNamesS = Ms.Properties.VariableNames(2:end);
baseS = Ms{1, seriesNamesS};
assert(~any(isnan(baseS)), 'SHORT panel: series NaN at Jun 2019, cannot rebase: %s', strjoin(seriesNamesS(isnan(baseS)), ', '));
Ms{:, seriesNamesS} = 100 * Ms{:, seriesNamesS} ./ baseS;

writetable(Ms, 'MasterData_USD_2019_2024_4country.xlsx');
% Saved as variable 'M' (not 'Ms') so every downstream script that does
% load('<file>.mat','M') only needs its filename changed, not its
% variable name, to switch between the long and short panels.
M = Ms;
save('MasterData_USD_2019_2024_4country.mat', 'M');

lvlS = table2array(Ms(:,2:end));
retS = diff(log(lvlS));
RS_ = array2table(retS, 'VariableNames', Ms.Properties.VariableNames(2:end));
RS_ = addvars(RS_, Ms.Period(2:end), 'Before', 1, 'NewVariableNames', 'Period');
writetable(RS_, 'MasterData_USD_2019_2024_4country_LogReturns.csv');

fprintf('\nSHORT/baseline panel (all 4 wine countries): %d months (%s to %s), %d series.\n', ...
    height(Ms), datestr(min(Ms.Period),'yyyy-mm'), datestr(max(Ms.Period),'yyyy-mm'), width(Ms)-1);
disp(Ms(1:3,:));
