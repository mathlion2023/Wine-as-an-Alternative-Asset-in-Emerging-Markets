%% AvgPrice_ByCountry.m (currency settled 2026-11-02)
% Quick addendum to Table 1: average price per bottle (equivalent, USD),
% by country, on the same Jan2013-Nov2024 window and same raw sources
% as DataCoverage_Liquidity_2019_2024.m.
%
% CURRENCY: Argentina, Australia, and Chile all come from Data
% Australia.xlsx, which Dr. Verdickt confirmed (2026-11-02) is uniformly
% AUD-denominated regardless of auction house or country -- converted here
% via normalize_lot_currency.m, the same function Build_RS_Hybrid_
% AllCountries.m uses. Reporting everything in USD (rather than native
% currency) is a deliberate choice, not an oversight: a native-currency
% column for Argentina/Chile would still mix genuinely different
% currencies (ARS/CLP) that this dataset does not actually carry at the
% lot level. South Africa (Strauss & Co., ZAR) is reported in USD too so
% all four countries are on the same, directly comparable basis in
% Table 1b, consistent with the USD basis used throughout the risk and
% portfolio analysis (Sections 4-5).
clear; clc;

% 2026-10-?? peer-review fix: this script's header comment always claimed
% the Jan2013-Nov2024 window used by DataCoverage_Liquidity_2019_2024.m,
% but no date filter was ever actually applied, so this script silently
% counted every row in the raw files regardless of date -- producing a
% trade count inconsistent with Table 1's (tab_coverage.csv) for Australia
% (76,674 vs 69,425) and Chile (496 vs 473). Fixed by applying the same
% DATE_START/DATE_END window Table 1 uses.
DATE_START = datetime(2013,1,1);
DATE_END   = datetime(2024,11,30);

results = table('Size',[0 5], 'VariableTypes', ["string","double","double","double","double"], ...
    'VariableNames', ["Country","MeanPrice","MedianPrice","StdPrice","N"]);

% South Africa -- PENDING: see note in Build_RS_Hybrid_AllCountries.m /
% CLAUDE.md on the "All" vs "SAOnly" sheet discrepancy in the updated
% workbook (SAOnly undercounts real SA producers; All includes non-SA
% producers with no nationality flag to separate them). Left on "SAOnly"
% (closest to the previous, working behavior) without the now-missing
% Country filter, pending a decision on how to classify producers.
% Strauss & Co. prices are ZAR; convert to USD via the monthly FX panel.
T0 = readtable('Strauss Wine All updated.xlsx','Sheet','SAOnly');
T0.ymDate = dateshift(T0.AuctionDate, 'start', 'month');

optsFX = detectImportOptions('Wine DATA.xlsx', 'Sheet', 'Sheet1');
optsFX.VariableNamesRange = 'A6';
optsFX.DataRange          = 'A7';
Tfx = readtable('Wine DATA.xlsx', optsFX);
Tfx.Properties.VariableNames{1} = 'Date';
Tfx.Period = dateshift(datetime(Tfx.Date), 'start', 'month');
fxKeysZAR = cellstr(datestr(Tfx.Period, 'yyyy-mm'));
fxMapZAR = containers.Map(fxKeysZAR, num2cell(Tfx.USDZAR));

ymStrSA = cellstr(datestr(T0.ymDate, 'yyyy-mm'));
usdzar = nan(height(T0), 1);
for i = 1:height(T0)
    if isKey(fxMapZAR, ymStrSA{i}), usdzar(i) = fxMapZAR(ymStrSA{i}); end
end
pZAR = double(T0.PricePer750ml);
pUSD = pZAR ./ usdzar;   % ZAR is local-per-USD quote
inWindow = T0.ymDate >= DATE_START & T0.ymDate <= DATE_END;
good = ~isnan(pUSD) & pUSD > 0 & inWindow;
results = [results; {"South Africa", mean(pUSD(good)), median(pUSD(good)), std(pUSD(good)), sum(good)}];

% Argentina, Australia, Chile: same lot-level normalization as the
% canonical Build_RS_Hybrid_AllCountries.m. As of 2026-11-02, Dr. Verdickt
% confirmed directly that every price in this file is AUD regardless of
% auction house or country, so normalize_lot_currency.m applies a single
% AUDUSD conversion to all rows (the earlier per-house EUR/GBP/ZAR
% inference this script used is superseded, not merely out of sync).
T1 = readtable('Data Australia.xlsx','Sheet','Sheet1');
T1.ymDate = datetime(T1.ym, 'InputFormat', 'yyyy-MM');
T1 = normalize_lot_currency(T1);

inWindow1 = T1.ymDate >= DATE_START & T1.ymDate <= DATE_END;
for cc = ["Argentina","Australia","Chile"]
    Tc = T1(strcmp(T1.wine_country, cc) & inWindow1, :);
    p = double(Tc.priceUSD); p = p(~isnan(p) & p>0);
    results = [results; {cc, mean(p), median(p), std(p), numel(p)}];
end

disp(results);
writetable(results, 'tab_avgprice_bycountry.csv');
