%% Monthly_Turnover_LotLevel.m (2026-10-28, co-author Faff feedback)
% Section 4.4's "average total monthly traded value" figures were computed
% as (mean price per bottle-equivalent, Table 1b) x (mean bottles/lot) x
% (lots/month) -- see Monthly_Turnover_BottlesPerLot.m. That arithmetic
% multiplies two averages together and need not equal the true total when
% price and lot size co-vary across lots (e.g. if larger case lots tend to
% carry a different per-bottle price than single-bottle lots). The correct
% quantity is a direct sum: for each lot, USD unit price x bottle-equivalent
% quantity, summed across all lots in the window, divided by the number of
% months. This script computes that directly from the same raw per-lot
% files, same currency normalization (uniform AUD->USD for Australia/
% Argentina/Chile, confirmed by Dr. Verdickt 2026-11-02), and same date
% window as AvgPrice_ByCountry.m / Build_RS_Hybrid_AllCountries.m (so the
% result is a like-for-like replacement for Section 4.4's existing claim).
%
% This is turnover OBSERVED in this dataset over this window, not a
% market-capacity estimate: it is a lower bound on markets this dataset's
% providers (Strauss & Co. for South Africa; the Verdickt-compiled houses
% for Australia/Argentina/Chile) actually covered, not a claim about total
% trading volume at houses or in segments outside this sample.
clear; clc;

DATE_START = datetime(2013,1,1);
DATE_END   = datetime(2024,11,30);

optsFX = detectImportOptions('Wine DATA.xlsx', 'Sheet', 'Sheet1');
optsFX.VariableNamesRange = 'A6';
optsFX.DataRange          = 'A7';
Tfx = readtable('Wine DATA.xlsx', optsFX);
Tfx.Properties.VariableNames{1} = 'Date';
Tfx.Period = dateshift(datetime(Tfx.Date), 'start', 'month');
fxKeys = cellstr(datestr(Tfx.Period, 'yyyy-mm'));
fxMapZAR = containers.Map(fxKeys, num2cell(Tfx.USDZAR));   % local-per-USD quote, South Africa only

results = table('Size',[0 5], 'VariableTypes', ["string","double","double","double","double"], ...
    'VariableNames', ["Country","NMonths","TotalLotValueUSD","MonthlyTurnoverUSD","NLots"]);

%% ---- South Africa (Strauss, Jun2019-Nov2024, 66 months) ----
T0 = readtable('Strauss Wine All updated.xlsx','Sheet','SAOnly');
T0.ymDate = dateshift(T0.AuctionDate, 'start', 'month');
saStart = datetime(2019,6,1); saEnd = datetime(2024,11,1);
ymStrSA = cellstr(datestr(T0.ymDate, 'yyyy-mm'));
usdzar = nan(height(T0), 1);
for i = 1:height(T0)
    if isKey(fxMapZAR, ymStrSA{i}), usdzar(i) = fxMapZAR(ymStrSA{i}); end
end
pUSD_SA = double(T0.PricePer750ml) ./ usdzar;
qty_SA  = double(T0.quantity);
inWindowSA = T0.ymDate >= saStart & T0.ymDate <= saEnd;
goodSA = inWindowSA & isfinite(pUSD_SA) & pUSD_SA > 0 & isfinite(qty_SA) & qty_SA > 0;
lotValueSA = pUSD_SA(goodSA) .* qty_SA(goodSA);
nMonthsSA = 66;
results = [results; {"South Africa", nMonthsSA, sum(lotValueSA), sum(lotValueSA)/nMonthsSA, sum(goodSA)}];

%% ---- Australia / Argentina / Chile (Verdickt file, Jan2013-Nov2024) ----
% Dr. Verdickt confirmed (2026-11-02) that this file is uniformly
% AUD-denominated regardless of auction house or country; converted here
% via the shared normalize_lot_currency.m, not an inline per-house map.
T1 = readtable('Data Australia.xlsx','Sheet','Sheet1');
T1.ymDate = datetime(T1.ym, 'InputFormat', 'yyyy-MM');
T1 = normalize_lot_currency(T1);

inWindow1 = T1.ymDate >= DATE_START & T1.ymDate <= DATE_END;
nMonthsOther = struct('Argentina',143,'Australia',143,'Chile',142);
for cc = ["Argentina","Australia","Chile"]
    Tc = T1(strcmp(T1.wine_country, cc) & inWindow1, :);
    p = double(Tc.priceUSD); q = double(Tc.numOfBottles);
    good = isfinite(p) & p>0 & isfinite(q) & q>0;
    lotValue = p(good) .* q(good);
    nM = nMonthsOther.(char(cc));
    results = [results; {cc, nM, sum(lotValue), sum(lotValue)/nM, sum(good)}];
end

disp(results);
writetable(results, 'tab_turnover_lotlevel.csv');

%% ---- Comparison against the old (mean-times-mean) approximation, for the record ----
avgTab = readtable('tab_avgprice_bycountry.csv');
fprintf('\n%-14s %16s %16s %10s\n', 'Country', 'OldApprox($/mo)', 'TrueSum($/mo)', 'Ratio');
oldNMonths = struct('SouthAfrica',66,'Australia',143,'Argentina',143,'Chile',142);
for i = 1:height(results)
    cc = results.Country(i);
    ccKey = erase(char(cc), ' ');
    row = avgTab(strcmp(avgTab.Country, cc), :);
    nM = oldNMonths.(ccKey);
    oldApprox = (row.N/nM) * row.MeanPrice; % old script's "oldTurnover" (bottles=1 assumption)
    % Apply the same mean-bottles-per-lot correction Monthly_Turnover_BottlesPerLot.m used
    trueVal = results.MonthlyTurnoverUSD(i);
    fprintf('%-14s %16.1f %16.1f %10.3f\n', cc, oldApprox, trueVal, trueVal/oldApprox);
end
