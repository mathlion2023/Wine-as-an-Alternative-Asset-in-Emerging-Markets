%% Build_RS_Hybrid_AllCountries.m (2026-09-03, currency fix 2026-10-08)
% -------------------------------------------------------------------------
% Rebuilds the pure repeat-sales (RS) and hybrid hedonic-repeat-sales (HY)
% price indices for all four countries on the extended Jan2013-Nov2024
% window, replacing the orphaned Repeatsales.mat/Projectdata.mat that no
% script in this repo actually produced (grep for save('Repeatsales' /
% save('Projectdata' across every .m file returns nothing -- those two
% .mat files were cached outputs of some undocumented, non-reproducible
% run, most likely on the old 2019-2024 window given the rest of the
% pipeline's history).
%
% Method: Fogarty & Jones (2011) hybrid hedonic-repeat-sales estimator,
% eqs. (1)-(6), as already partially implemented in ../IndexConstruction.m
% -- that script's core GLS logic (steps 2-7 there) is reused here nearly
% verbatim, parameterized by country instead of hardcoded, and FIXED: the
% original was headed "HybridAustralianWineIndex.m" / titled "Australian
% Fine Wine" throughout, but its actual country filter was
% strcmp(wine_country,'Argentina') and most of its output filenames said
% ARG -- an incomplete copy-paste from an Australia template to build an
% Argentina version. That mislabeling is not reproduced here: each
% country gets its own explicit, checked filter and output name.
%
% The pure RS index is NOT a separate model -- it falls out of the same
% pipeline as an intermediate quantity: the repeat-sales-only differenced
% regression (b_rs, step 5a) IS the pure repeat-sales index in log-diff
% form; the paper's Algorithm 1 validation step ("benchmark against pure
% hedonic and pure RS indices") is exactly this. No extra estimation is
% needed, just extracting what the hybrid GLS already computes on the way
% to the pooled estimate.
% -------------------------------------------------------------------------
clear; clc;

startDate = datetime(2013,1,1);
endDate   = datetime(2024,11,1);

% Canonical monthly grid, reused from the master USD dataset so this
% output is directly comparable/mergeable with everything else in the
% pipeline (same 143-month grid as MasterData_USD_2013_2024.mat).
load('MasterData_USD_2013_2024.mat', 'M');
grid = M.Period;
% BUG FIX (2026-10-09): stale assert expected 143 months (an earlier
% Jan2013 start); the panel correctly starts Feb2013 per the manuscript
% (142 months) -- MasterData_USD_2013_2024.mat has been 142 months since
% the LONG panel was finalized, this assert was simply never updated.
assert(numel(grid) == 142, 'Expected 142-month grid, got %d.', numel(grid));

countries = struct( ...
    'key',   {'SAFW', 'AUSFW', 'ARGFW', 'CHIFW'}, ...
    'source',{'strauss','verdickt','verdickt','verdickt'}, ...
    'filter',{'', 'Australia', 'Argentina', 'Chile'});

results = struct();

%% ---- Load Verdickt file once (AUS/ARG/CHI share it) ----
Tver = readtable('Data Australia.xlsx','Sheet','Sheet1');
Tver.ymDate = datetime(Tver.ym, 'InputFormat', 'yyyy-MM');

% CURRENCY, settled 2026-11-02: three prior rounds (2026-10-08 Langton's
% vs. USD; 2026-10-22 per-house EUR/GBP/ZAR map; 2026-10-29 a further
% six-currency FX extension plus a quantified, disclosed Acker/Sotheby's/
% Christie's/Bonhams ambiguity) all inferred currency from each auction
% house's jurisdiction, because Data Australia.xlsx carries no
% sale-location field and Dr. Gertjan Verdickt's own records had not yet
% been checked directly. On direct request, Verdickt confirmed that every
% price in this file -- every auction house, every country (Australia,
% Argentina, Chile) -- is denominated in Australian dollars. All lots now
% convert via the same monthly AUDUSD rate (normalize_lot_currency.m); the
% house-by-house inference is superseded, not merely one option among
% several.
Tver = normalize_lot_currency(Tver);
house = string(Tver.auction_house);
fprintf('Currency normalization applied via normalize_lot_currency.m (uniform AUD->USD via AUDUSD, confirmed directly by Dr. Verdickt 2026-11-02) to %d rows.\n', height(Tver));

%% ---- Load Strauss file once (SA) ----
% The updated workbook's "SAOnly" sheet no longer carries a Country
% column (it dropped it as of the 2026-09-04 update) -- the sheet is used
% as-is since it is already Strauss & Co.'s curated South-Africa listing.
Tsa = readtable('Strauss Wine All updated.xlsx','Sheet','SAOnly');
Tsa.ymDate = dateshift(Tsa.AuctionDate, 'start', 'month');

for c = 1:numel(countries)
    key = countries(c).key;
    fprintf('\n=== %s ===\n', key);

    if strcmp(countries(c).source, 'verdickt')
        Tc = Tver(strcmp(Tver.wine_country, countries(c).filter), :);
        Tc = Tc(Tc.ymDate >= startDate & Tc.ymDate <= endDate, :);
        name    = string(Tc.name);
        vintage = Tc.vintage;
        ymDate  = Tc.ymDate;
        price   = Tc.priceUSD;  % already USD-normalized at the lot level (see currency fix above)
    else % strauss / SA
        Tc = Tsa(Tsa.ymDate >= startDate & Tsa.ymDate <= endDate, :);
        name    = string(Tc.WineName);
        vintage = Tc.Vintage;
        ymDate  = Tc.ymDate;
        price   = Tc.PricePer750ml;
    end

    good = isfinite(price) & price > 0 & ~isnat(ymDate) & isfinite(vintage);
    name = name(good); vintage = vintage(good); ymDate = ymDate(good); price = price(good);
    fprintf('%s: %d lot-level rows in window after cleaning (native range %s to %s).\n', ...
        key, numel(price), datestr(min(ymDate)), datestr(max(ymDate)));

    % BUG FIX (2026-09-03): South Africa's label dimension (1,223 distinct
    % WineName values over ~3,700 cells, most seen once or twice -- the
    % same sparsity already documented in the Hedonic Regression appendix,
    % which for exactly this reason drops the label effect for South
    % Africa and reports time+vintage only) makes the hedonic block's
    % full name-dummy design singular (RCOND=NaN in the first attempt at
    % this script, producing an all-NaN hybrid index). Use time+vintage
    % only for South Africa's hedonic block, consistent with the already-
    % established finding in Appendix B, not a new ad hoc choice.
    useNameDummies = ~strcmp(key, 'SAFW');
    % Request the second output (2026-10-09, peer-review item 3): item-
    % clustered sandwich SEs for the time-dummy coefficients, so Section
    % 4.2's claim of "uncertainty quantified via heteroskedasticity-robust
    % (sandwich) standard errors" is backed by an actual computation
    % rather than asserted with nothing behind it.
    [R, se_hy] = fogartyJonesHybridRS(name, vintage, ymDate, log(price), useNameDummies);
    R.Period = dateshift(R.Period, 'start', 'month');
    R.SE_LogIndex_HY = se_hy;

    % BUG FIX (2026-09-03): interpolating with 'extrap' silently
    % fabricated South African index values for 2013-2018 -- Strauss &
    % Co. (the SA data source) was only founded in 2019, so there is
    % genuinely no SA data before June 2019; the first attempt at this
    % script linearly extrapolated backward from the 2019-2024 trend,
    % producing an implausible ~3,450-level index in January 2013 with no
    % basis in any real transaction. Do NOT extrapolate for any country:
    % interpolate strictly within each country's own observed native
    % range, leaving NaN (not a fabricated value) outside it.
    logHY_grid = interp1(datenum(R.Period), R.LogIndex_HY, datenum(grid), 'linear');
    logRS_grid = interp1(datenum(R.Period), R.LogIndex_RS, datenum(grid), 'linear');

    results.(key).HY_native = R.LogIndex_HY;
    results.(key).RS_native = R.LogIndex_RS;
    results.(key).Period_native = R.Period;
    results.(key).HY = exp(logHY_grid) * 100; % base=100 convention, matching the rest of the pipeline
    results.(key).RS = exp(logRS_grid) * 100;
    results.(key).nCells = R.nCellsUsed(1);
    results.(key).nPairs = R.nPairsUsed(1);
    results.(key).nativeStart = min(ymDate);
    results.(key).nativeEnd = max(ymDate);
    % Item-clustered sandwich SE of the log-index, delta-method-mapped to
    % the level index (SE_level ~ level * SE_log); interpolated onto the
    % same grid as HY/RS (NaN outside the native window, same policy).
    results.(key).SE_LogIndex_HY_native = R.SE_LogIndex_HY;
    seLog_grid = interp1(datenum(R.Period), R.SE_LogIndex_HY, datenum(grid), 'linear');
    results.(key).SE_HY = results.(key).HY .* seLog_grid;
    fprintf('%s: median monthly log-index SE = %.4f (range %.4f-%.4f), i.e. a typical +/-1SE band of roughly +/-%.0f%% on the index level.\n', ...
        key, median(R.SE_LogIndex_HY,'omitnan'), min(R.SE_LogIndex_HY), max(R.SE_LogIndex_HY), 100*median(R.SE_LogIndex_HY,'omitnan'));
end

%% ---- Assemble into one table, explicit named columns (no positional indexing) ----
T = table(grid, 'VariableNames', {'Period'});
for c = 1:numel(countries)
    key = countries(c).key;
    T.(key + "_HY") = results.(key).HY(:);
    T.(key + "_RS") = results.(key).RS(:);
    T.(key + "_SE_HY") = results.(key).SE_HY(:);
end

% NOTE on currency: AUSFW/ARGFW/CHIFW are now natively USD (normalized at
% the lot level above, before index construction) -- do NOT apply any
% further AUDUSD/USDARS/USDCLP conversion downstream. SAFW is still in
% local currency (ZAR, Strauss & Co.'s own pricing) and still needs the
% usual usd(level, USDZAR) conversion downstream.
writetable(T, 'RS_Hybrid_Comparison_18.csv');
save('RS_Hybrid_Comparison_18.mat', 'T', 'results', 'countries');
fprintf('\nSaved RS_Hybrid_Comparison_18.csv / .mat (AUSFW/ARGFW/CHIFW already USD; SAFW still ZAR, convert downstream)\n');
disp(T(1:5,:));


%% ============================= FUNCTIONS ================================
% fogartyJonesHybridRS extracted (2026-10-09) to its own file,
% fogartyJonesHybridRS.m, so Portfolio_Backtest_PointInTime_4country.m
% (peer-review item 6: no-lookahead backtest) can call the identical
% estimator on truncated data without duplicating it. No logic change.
