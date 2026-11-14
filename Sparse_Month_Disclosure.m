%% Sparse_Month_Disclosure.m (2026-10-09, peer-review-flagged)
% -------------------------------------------------------------------------
% Peer review (concern #2) flagged that Table 1's ">=8 trades" threshold is
% never stated to be a REPORTING cutoff only -- it does not gate which
% months enter the hybrid hedonic-repeat-sales index. In fact (confirmed by
% reading Build_RS_Hybrid_AllCountries.m / fogartyJonesHybridRS.m): every
% calendar month with >=1 trade gets its own hedonic time-dummy coefficient
% (timeLevels = categories(categorical(Tagg.ym))), and whether that month
% ALSO has repeat-sales-pair identification depends on whether any
% name+vintage item traded in that month and at least one other month.
% Calendar months with ZERO trades are not estimated at all -- they are
% linearly interpolated within each country's native window (never
% extrapolated outside it).
%
% This script reproduces the raw-data loading/filtering of
% Build_RS_Hybrid_AllCountries.m (same files, same cleaning, same currency
% treatment -- read-only, no re-estimation) purely to COUNT, per country:
%   - total calendar months in the native window (first to last trade)
%   - months with >=8 trades (Table 1's existing reporting threshold)
%   - months with 1-7 trades (qualify for hedonic estimation, not for
%     Table 1's threshold)
%   - months with >=1 trade that have NO repeat-sales connectivity (i.e.
%     every item traded that month is a one-off within the window --
%     identified by hedonic cross-section only, not pinned by any
%     same-item repeat sale)
%   - months with zero trades (interpolated, not estimated)
% -------------------------------------------------------------------------
clear; clc;

startDate = datetime(2013,1,1);
endDate   = datetime(2024,11,1);

countries = struct( ...
    'key',   {'SAFW', 'AUSFW', 'ARGFW', 'CHIFW'}, ...
    'source',{'strauss','verdickt','verdickt','verdickt'}, ...
    'filter',{'', 'Australia', 'Argentina', 'Chile'});

Tver = readtable('Data Australia.xlsx','Sheet','Sheet1');
Tver.ymDate = datetime(Tver.ym, 'InputFormat', 'yyyy-MM');

% CURRENCY: Dr. Verdickt confirmed (2026-11-02) this source file is
% uniformly AUD-denominated regardless of auction house or country, so
% every row converts via the shared normalize_lot_currency.m, matching
% Build_RS_Hybrid_AllCountries.m exactly.
Tver = normalize_lot_currency(Tver);
fprintf('Sparse_Month_Disclosure.m: currency-converted all %d rows via normalize_lot_currency.m (uniform AUD->USD), matching Build_RS_Hybrid_AllCountries.m.\n', height(Tver));

Tsa = readtable('Strauss Wine All updated.xlsx','Sheet','SAOnly');
Tsa.ymDate = dateshift(Tsa.AuctionDate, 'start', 'month');

rows = {};
for c = 1:numel(countries)
    key = countries(c).key;

    if strcmp(countries(c).source, 'verdickt')
        Tc = Tver(strcmp(Tver.wine_country, countries(c).filter), :);
        Tc = Tc(Tc.ymDate >= startDate & Tc.ymDate <= endDate, :);
        name    = string(Tc.name);
        vintage = Tc.vintage;
        ymDate  = Tc.ymDate;
        price   = Tc.priceUSD;  % BUG FIX finding #7: was Tc.pricePerBottle (no currency conversion)
    else
        Tc = Tsa(Tsa.ymDate >= startDate & Tsa.ymDate <= endDate, :);
        name    = string(Tc.WineName);
        vintage = Tc.Vintage;
        ymDate  = Tc.ymDate;
        price   = Tc.PricePer750ml;
    end

    good = isfinite(price) & price > 0 & ~isnat(ymDate) & isfinite(vintage);
    name = name(good); vintage = vintage(good); ymDate = ymDate(good);

    nativeStart = dateshift(min(ymDate), 'start', 'month');
    nativeEnd   = dateshift(max(ymDate), 'start', 'month');
    fullGrid = (nativeStart:calmonths(1):nativeEnd)';
    nTotalMonths = numel(fullGrid);

    monthOf = dateshift(ymDate, 'start', 'month');
    [uMonths, ~, ic] = unique(monthOf);
    tradeCount = accumarray(ic, 1);
    countByMonth = containers.Map(cellstr(datestr(uMonths,'yyyy-mm')), num2cell(tradeCount));

    % Item ID = name + vintage, same as fogartyJonesHybridRS's grouping key
    ID = strcat(name, "__", string(vintage));
    [uID, ~, ig] = unique(ID);
    nGroups = numel(uID);
    rsConnectedMonths = false(0,1);
    rsMonthSet = containers.Map('KeyType','char','ValueType','logical');
    for g = 1:nGroups
        idx = (ig == g);
        if sum(idx) < 2, continue; end
        mg = unique(monthOf(idx));
        for m = 1:numel(mg)
            rsMonthSet(datestr(mg(m),'yyyy-mm')) = true;
        end
    end

    nGe8 = 0; n1to7 = 0; nZero = 0; nHedonicOnly = 0;
    for m = 1:nTotalMonths
        kstr = datestr(fullGrid(m),'yyyy-mm');
        if isKey(countByMonth, kstr)
            cnt = countByMonth(kstr);
            if cnt >= 8
                nGe8 = nGe8 + 1;
            else
                n1to7 = n1to7 + 1;
            end
            if ~isKey(rsMonthSet, kstr)
                nHedonicOnly = nHedonicOnly + 1;
            end
        else
            nZero = nZero + 1;
        end
    end

    fprintf('%s: %d total months | %d >=8 trades | %d 1-7 trades | %d hedonic-only (no RS pair) | %d zero-trade (interpolated)\n', ...
        key, nTotalMonths, nGe8, n1to7, nHedonicOnly, nZero);

    rows(end+1,:) = {key, nTotalMonths, nGe8, n1to7, nHedonicOnly, nZero}; %#ok<AGROW>
end

T = cell2table(rows, 'VariableNames', ...
    {'Country','TotalMonths','MonthsGe8Trades','Months1to7Trades','MonthsHedonicOnlyNoRS','MonthsZeroTradeInterpolated'});
disp(T);
writetable(T, 'tab_sparse_month_disclosure.csv');
fprintf('Saved tab_sparse_month_disclosure.csv\n');
