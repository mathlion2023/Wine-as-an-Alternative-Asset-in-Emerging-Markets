%% Build_MasterData_USD.m
% -------------------------------------------------------------------------
% Consolidates the updated 2013-01 to 2024-11 wine-index panel
% (EmergingCountryIndex.xlsx) with the Bloomberg benchmark/FX panel
% (Wine DATA.xlsx, pulled per Bloomberg_Index_Tickers_Wine_Revision.xlsx)
% into a single monthly, USD-denominated master dataset for the
% portfolio/spillover pipeline (Referee 2's local-equity-benchmark
% request; Referee 1's Liv-ex-comparability point).
%
% CURRENCY ASSUMPTION (not stated in the source files -- confirm with
% Strauss & Co. / Dr. Verdickt before citing in the manuscript):
%   South Africa wine index  -> ZAR (Strauss & Co., Cape Town auctions)
%   Australia wine index     -> AUD
%   Argentina wine index     -> ARS
%   Chile wine index         -> CLP
%
% Output:
%   MasterData_USD_2013_2024.xlsx / .mat  - USD levels, monthly
%   MasterData_USD_LogReturns.csv         - USD log-returns, monthly
% -------------------------------------------------------------------------
%
% *** SUPERSEDED (2026-10-08) -- DO NOT RUN ***
% This v1 script builds the wine-index panel from EmergingCountryIndex.xlsx,
% which was found to carry unreliable/mismatched South Africa data back to
% 2013 (Strauss & Co., SA's only real data source, was founded 2019-06) and
% to apply a country-wide FX conversion that mixes currencies incorrectly
% for Argentina/Chile lots that actually clear in USD. It writes to the
% SAME output filenames (MasterData_USD_2013_2024.xlsx/.mat and
% MasterData_USD_LogReturns.csv) as Build_MasterData_USD_v2.m, which fixes
% both issues via the lot-level hybrid-index pipeline (fogartyJonesHybridRS.m
% / Build_RS_Hybrid_AllCountries.m). Running this script would silently
% overwrite the corrected panel every downstream script loads. Use
% Build_MasterData_USD_v2.m instead. (Code-review finding #3, 2026-10-10.)
error('Build_MasterData_USD.m is superseded by Build_MasterData_USD_v2.m -- see header comment. Aborting to avoid overwriting the corrected master dataset.');

clear; clc;

%% ---- 1. Wine indices (Jan 2013 - Nov 2024), local currency, base=100 @ 2013-01 ----
Twine = readtable('EmergingCountryIndex.xlsx', 'Sheet', 'Sheet1');
Twine.Properties.VariableNames = {'Period','ARG','AUS','CHL_raw','SA','CHL'};

% CHL_raw (col D) and CHL (col F) are the SAME Chile series at two
% different bases -- confirmed CHL_raw(t)/CHL(t) is constant (~1.3442),
% not two distinct countries. CHL (base=100 @ 2013-01) matches the
% convention used by ARG/AUS/SA, so that is the one we keep.
ratioCheck = Twine.CHL_raw ./ Twine.CHL;
assert(range(ratioCheck) < 1e-6, ...
    'EmergingCountryIndex.xlsx: CHL_raw/CHL is no longer a constant ratio -- these may now be two different series. Re-check before dropping CHL_raw.');
Twine.CHL_raw = [];
Twine = movevars(Twine, 'CHL', 'After', 'AUS');
% Twine columns: Period | ARG | AUS | CHL | SA   (all local currency, base100 @ 2013-01)

Twine.Period = datetime(Twine.Period);
Twine.Period = dateshift(Twine.Period, 'start', 'month');

%% ---- 2. Bloomberg benchmark + FX panel ----
% Header layout: row1=Group, row2=Series, row3=Ticker, row4=Field,
% row5=Currency, row6=machine-friendly code (Date, SPTR_TR, ...), data row7+.
optsBB = detectImportOptions('Wine DATA.xlsx', 'Sheet', 'Sheet1');
optsBB.VariableNamesRange = 'A6';
optsBB.DataRange          = 'A7';
Tbb = readtable('Wine DATA.xlsx', optsBB);
Tbb.Properties.VariableNames{1} = 'Date';
Tbb.Date = datetime(Tbb.Date);
Tbb.Period = dateshift(Tbb.Date, 'start', 'month');

% Restrict to the wine-index sample window
Tbb = Tbb(Tbb.Period >= min(Twine.Period) & Tbb.Period <= max(Twine.Period), :);

%% ---- 3. Merge on Period ----
T = innerjoin(Twine, Tbb, 'Keys', 'Period');
assert(height(T) == height(Twine), ...
    'Merge dropped rows: Wine DATA.xlsx does not fully cover the wine-index sample window (Jan 2013-Nov 2024).');

%% ---- 4. Convert everything to USD ----
% Bloomberg FX convention: USDZAR/USDARS/USDCLP are quoted "local units
% per USD" (divide local level by the rate to get USD); AUDUSD/EURUSD/
% GBPUSD are quoted "USD per foreign unit" (multiply local level by the
% rate to get USD).
usd = @(localLevel, fx) localLevel ./ fx;   % local-per-USD quote
usdM = @(localLevel, fx) localLevel .* fx;  % USD-per-local quote

M = table();
M.Period = T.Period;

% Wine indices
M.SA_wine_USD  = usd(T.SA,  T.USDZAR);
M.AUS_wine_USD = usdM(T.AUS, T.AUDUSD);
M.ARG_wine_USD = usd(T.ARG, T.USDARS);
M.CHL_wine_USD = usd(T.CHL, T.USDCLP);

% Liv-ex (GBP). LivEx50/LivExInv dropped: shorter history than the rest
% of the panel (N_eff 120/61 of 142 vs. 142 for every other series) --
% only Liv-ex 100 has full Jan 2013-Nov 2024 coverage.
M.LivEx100_USD = usdM(T.LIVEX100, T.GBPUSD);

% Domestic equity benchmarks (Referee 2)
M.JALSH_USD  = usd(T.JALSH_TR,  T.USDZAR);
M.ASX200_USD = usdM(T.ASX200_TR, T.AUDUSD);
M.MERVAL_USD = usd(T.MERVAL_TR, T.USDARS);
M.IPSA_USD   = usd(T.IPSA_TR,   T.USDCLP);

% Established-Europe equity benchmarks (EUR)
M.CAC_USD  = usdM(T.CAC_TR,  T.EURUSD);
M.MIB_USD  = usdM(T.MIB_TR,  T.EURUSD);
M.IBEX_USD = usdM(T.IBEX_TR, T.EURUSD);
% PSI: prefer the gross total-return field; fall back to price if that
% column is not populated for this vintage of the download.
if all(isnan(T.PSI_TR))
    warning('PSI_TR is empty in this Wine DATA.xlsx download; falling back to PSI_PX (price index, not total return).');
    M.PSI_USD = usdM(T.PSI_PX, T.EURUSD);
else
    M.PSI_USD = usdM(T.PSI_TR, T.EURUSD);
end
M.DAX_USD = usdM(T.DAX, T.EURUSD);

% Already USD
M.SPX_USD      = T.SPTR_TR;
M.MSCIEM_USD   = T.MXEF_TR;
M.GlobalAgg_USD = T.GLOBAGG_TR;
M.Gold_USD     = T.GOLD;

%% ---- 5. Keep only fully-covered series (143 months, no gaps) ----
seriesNames0 = M.Properties.VariableNames(2:end);
nValid = sum(~isnan(M{:, seriesNames0}), 1);
short = seriesNames0(nValid < height(M));
if ~isempty(short)
    warning('Dropping series with <%d months of data: %s', height(M), strjoin(short, ', '));
    M = removevars(M, short);
end

%% ---- 6. Rebase every series to 100 @ Jan 2013 ----
% Levels are converted USD figures with no shared scale (wine indices
% were already ~100-based in local currency; Bloomberg equity/FX levels
% are not), so rebasing puts every series on the same comparable footing.
assert(all(M.Period(1) == datetime(2013,1,1)), ...
    'First row of the master dataset is not Jan 2013 -- check the merge/date alignment before rebasing.');
seriesNames = M.Properties.VariableNames(2:end);
base = M{1, seriesNames};
assert(~any(isnan(base)), ...
    'At least one series is NaN at Jan 2013, so it cannot be rebased to 100 there: %s', ...
    strjoin(seriesNames(isnan(base)), ', '));
M{:, seriesNames} = 100 * M{:, seriesNames} ./ base;

%% ---- 7. Save levels ----
writetable(M, 'MasterData_USD_2013_2024.xlsx');
save('MasterData_USD_2013_2024.mat', 'M');

%% ---- 8. Log-returns (feeds descriptiveStats / vol-model pipeline) ----
lvl = table2array(M(:,2:end));
ret = diff(log(lvl));
R = array2table(ret, 'VariableNames', M.Properties.VariableNames(2:end));
R = addvars(R, M.Period(2:end), 'Before', 1, 'NewVariableNames', 'Period');
writetable(R, 'MasterData_USD_LogReturns.csv');

fprintf('Master USD dataset: %d months (%s to %s), %d series.\n', ...
    height(M), datestr(min(M.Period),'yyyy-mm'), datestr(max(M.Period),'yyyy-mm'), width(M)-1);
disp(M(1:3,:));
