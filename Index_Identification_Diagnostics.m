%% Index_Identification_Diagnostics.m (2026-10-09, peer-review item 2)
% -------------------------------------------------------------------------
% Peer review point 2 asks: is the 8-trade threshold actually enforced (no
% -- Section 3.4 already states this and Table 1a documents interpolation
% of zero-trade months); how are insufficient-activity/zero-trade months
% handled (interpolation, already documented); and -- not previously
% checked anywhere in this repo -- is the hybrid hedonic-repeat-sales
% design matrix actually full rank, and does the repeat-sales "trading
% graph" (nodes = months with >=1 trade, edges = a repeat-sales pair
% linking two months) connect every estimated month back to the base
% month? Disconnected months, or rank deficiency, would mean some time
% coefficients are not actually identified by the data, however
% plausible-looking the output.
%
% This is a READ-ONLY diagnostic: it reproduces Build_RS_Hybrid_AllCountries.m's
% data loading and cell-aggregation exactly (same filters, same currency
% fix, same window), then checks rank and graph connectivity on the
% resulting design, WITHOUT modifying or re-deriving the production
% index itself. Kept as a separate script rather than edited into
% Build_RS_Hybrid_AllCountries.m so the already-verified Table 1/3
% pipeline is not touched by this diagnostic pass.
% -------------------------------------------------------------------------
clear; clc;

startDate = datetime(2013,1,1);
endDate   = datetime(2024,11,1);

countries = struct( ...
    'key',   {'SAFW', 'AUSFW', 'ARGFW', 'CHIFW'}, ...
    'source',{'strauss','verdickt','verdickt','verdickt'}, ...
    'filter',{'', 'Australia', 'Argentina', 'Chile'});

%% ---- Load Verdickt file once (AUS/ARG/CHI share it) ----
Tver = readtable('Data Australia.xlsx','Sheet','Sheet1');
Tver.ymDate = datetime(Tver.ym, 'InputFormat', 'yyyy-MM');

% CURRENCY: Dr. Verdickt confirmed (2026-11-02) this source file is
% uniformly AUD-denominated regardless of auction house or country, so
% every row converts via the shared normalize_lot_currency.m (matching
% Build_RS_Hybrid_AllCountries.m exactly, per this script's own header).
Tver = normalize_lot_currency(Tver);

%% ---- Load Strauss file once (SA) ----
Tsa = readtable('Strauss Wine All updated.xlsx','Sheet','SAOnly');
Tsa.ymDate = dateshift(Tsa.AuctionDate, 'start', 'month');

results = table();
for c = 1:numel(countries)
    key = countries(c).key;
    fprintf('\n=== %s ===\n', key);

    if strcmp(countries(c).source, 'verdickt')
        Tc = Tver(strcmp(Tver.wine_country, countries(c).filter), :);
        Tc = Tc(Tc.ymDate >= startDate & Tc.ymDate <= endDate, :);
        name    = string(Tc.name);
        vintage = Tc.vintage;
        ymDate  = Tc.ymDate;
        price   = Tc.priceUSD;
    else
        Tc = Tsa(Tsa.ymDate >= startDate & Tsa.ymDate <= endDate, :);
        name    = string(Tc.WineName);
        vintage = Tc.Vintage;
        ymDate  = Tc.ymDate;
        price   = Tc.PricePer750ml;
    end

    good = isfinite(price) & price > 0 & ~isnat(ymDate) & isfinite(vintage);
    name = name(good); vintage = vintage(good); ymDate = ymDate(good); price = price(good);
    useNameDummies = ~strcmp(key, 'SAFW');

    diag = checkIdentification(name, vintage, ymDate, log(price), useNameDummies);
    diag.Country = string(key);
    results = [results; struct2table(diag)]; %#ok<AGROW>

    fprintf(['%s: %d cells, %d periods, %d RS pairs | hedonic design %dx%d, rank=%d ', ...
        '(deficiency=%d) | RS-graph: %d components (base-month component size=%d/%d traded months) ', ...
        '| pure-RS system %dx%d, rank=%d (deficiency=%d)\n'], ...
        key, diag.nCells, diag.nPeriods, diag.nPairs, diag.n, diag.Kh, diag.rankXh, diag.Kh-diag.rankXh, ...
        diag.nComponents, diag.baseComponentSize, diag.nTradedMonths, diag.TrsDim, diag.TrsDim, diag.rankDrs, diag.TrsDim-diag.rankDrs);
end

results = results(:, [end, 1:end-1]); % Country first
writetable(results, 'tab_index_identification_diagnostics.csv');
disp(results);
fprintf('\nSaved tab_index_identification_diagnostics.csv\n');

%% ============================= FUNCTIONS ================================
function out = checkIdentification(name, vintage, ymDate, logPrice, useNameDummies)
    ym = string(datestr(ymDate, 'yyyy-mm'));

    [G0, aggName, aggVint, aggYm] = findgroups(name, vintage, ym);
    aggLogP  = splitapply(@mean,  logPrice, G0);
    Tagg = table(aggName, aggVint, aggYm, aggLogP, 'VariableNames', {'name','vintage','ym','logP'});
    n = height(Tagg);

    timeCat    = categorical(Tagg.ym);
    timeLevels = categories(timeCat);
    nT         = numel(timeLevels);
    timeCode   = double(timeCat);
    D_time     = dummyvar(timeCat); D_time(:,1) = [];

    if useNameDummies
        nameCat = categorical(Tagg.name);
        D_name  = dummyvar(nameCat); D_name(:,1) = [];
    else
        D_name = zeros(n,0);
    end
    vintCat = categorical(string(Tagg.vintage));
    D_vint  = dummyvar(vintCat); D_vint(:,1) = [];

    Xh = [ones(n,1), D_name, D_vint, D_time];
    Kh = size(Xh,2);
    startIdx = 1 + size(D_name,2) + size(D_vint,2) + 1;

    % Rank of the hedonic design (full stacked hybrid system's rank can only
    % be >= this, since adding RS equations never removes identification --
    % so Xh's own rank deficiency is itself proof of a non-identified
    % hedonic time/vintage/label effect, independent of the RS graph check)
    rankXh = rank(Xh);

    % RS pairs (same construction as fogartyJonesHybridRS)
    Tagg.ID = strcat(Tagg.name, "__", string(Tagg.vintage));
    G = findgroups(Tagg.ID);
    nGroups = max(G);
    pairEarly = cell(nGroups,1); pairLate = cell(nGroups,1);
    for g = 1:nGroups
        idx = find(G == g);
        if numel(idx) < 2, continue; end
        [~, ord] = sort(timeCode(idx));
        idxSorted = idx(ord);
        combos = nchoosek(1:numel(idxSorted), 2);
        pairEarly{g} = idxSorted(combos(:,1));
        pairLate{g}  = idxSorted(combos(:,2));
    end
    i1 = cat(1, pairEarly{:}); i2 = cat(1, pairLate{:});
    nPairs = numel(i1);
    p1 = timeCode(i1); p2 = timeCode(i2);

    % Trading graph: nodes = distinct time periods with >=1 cell; edges =
    % a repeat-sales pair linking two periods. Includes self-loops when a
    % pair's two cells fall in the same calendar month (p1==p2); conncomp
    % ignores self-loops so these don't bias connectivity.
    tradedMonths = unique(timeCode);
    nTradedMonths = numel(tradedMonths);
    noSelfLoop = p1 ~= p2;
    Gph = graph(p1(noSelfLoop), p2(noSelfLoop), [], nT); % nT nodes = all time levels; untraded ones are isolated by construction
    comp = conncomp(Gph);
    % Base month = period 1 (the normalization g_hy(1)=0 / first time level)
    baseComp = comp(1);
    baseComponentSize = sum(comp(tradedMonths) == baseComp);
    nComponents = numel(unique(comp(tradedMonths))); % components among TRADED months only

    % Pure-RS system rank (same AtA as fogartyJonesHybridRS step 5)
    rows = (1:nPairs)';
    keepLate = p2 > 1; keepEarly = p1 > 1;
    colLate  = startIdx - 2 + p2; colEarly = startIdx - 2 + p1;
    Drs = sparse([rows(keepLate); rows(keepEarly)], ...
                 [colLate(keepLate); colEarly(keepEarly)], ...
                 [ones(sum(keepLate),1); -ones(sum(keepEarly),1)], nPairs, Kh);
    Drs_slim = Drs(:, startIdx:end);
    TrsDim = size(Drs_slim,2);
    % BUG FIX (2026-10-10, code-review finding #14): rank used to be computed
    % as rank(full(Drs_slim'*Drs_slim)), which squares Drs_slim's singular
    % values (and so its condition number) before rank()'s tolerance (scaled
    % to the largest singular value) is applied -- numerically noisier than
    % necessary for a diagnostic that exists specifically to catch rank
    % deficiency reliably (it already found a genuine one, for Chile, via
    % the separate rankXh computation above). Compute rank directly off
    % Drs_slim instead.
    rankDrs = rank(full(Drs_slim));

    out = struct('n',n, 'nCells',n, 'nPeriods',nT, 'nPairs',nPairs, ...
        'Kh',Kh, 'rankXh',rankXh, 'nTradedMonths',nTradedMonths, ...
        'nComponents',nComponents, 'baseComponentSize',baseComponentSize, ...
        'TrsDim',TrsDim, 'rankDrs',rankDrs);
end
