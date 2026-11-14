%% Monthly_Turnover_BottlesPerLot.m (2026-10-22, co-author Faff pre-submission close-out)
% Section 4.4 states average total monthly traded value of roughly
% $102,400 (Australia), $8,500 (South Africa), $280 (Chile), and $100
% (Argentina), derived as (lots/month, from Table 1) x (mean price per
% bottle, from Table 1b = tab_avgprice_bycountry.csv). That arithmetic
% silently treats every auction LOT as exactly one bottle. The raw data
% carries an explicit bottle-count field per lot (numOfBottles for the
% Verdickt AUS/ARG/CHI file, quantity for the Strauss SA file) that was
% never applied here. Mean bottles/lot is far from 1 for three of four
% countries (Argentina 4.57, Chile 4.15, South Africa 4.63 -- all driven
% by case-lot sales -- vs. Australia's more modest 2.28, still 37% of
% lots multi-bottle), so the stated turnover figures understate true
% monthly dollar volume by roughly 2.3x-4.6x, country-dependent. This
% script recomputes the correct figures using the same window and price
% conventions as AvgPrice_ByCountry.m / tab_avgprice_bycountry.csv (one
% call site per country: Australia/Argentina/Chile on the Jan2013-
% Nov2024 142/143-month Verdickt window, South Africa on the Jun2019-
% Nov2024 66-month Strauss window), so the corrected number is a direct,
% like-for-like replacement for the text's existing claim, not a new
% methodology.
clear; clc;

%% South Africa (Strauss, 66-month window; mean price already in USD per tab_avgprice_bycountry.csv)
Tsa = readtable('Strauss Wine All updated.xlsx','Sheet','SAOnly');
Tsa.ymDate = dateshift(Tsa.AuctionDate, 'start', 'month');
startDate = datetime(2019,6,1); endDate = datetime(2024,11,1);
Tsa = Tsa(Tsa.ymDate >= startDate & Tsa.ymDate <= endDate, :);
nMonthsSA = 66;
meanBottlesSA = mean(Tsa.quantity, 'omitnan');

%% Australia / Argentina / Chile (Verdickt, Jan2013-Nov2024 window)
Tv = readtable('Data Australia.xlsx','Sheet','Sheet1');
Tv.ymDate = datetime(Tv.ym, 'InputFormat', 'yyyy-MM');

avgTab = readtable('tab_avgprice_bycountry.csv'); % Table 1b: MeanPrice, N, by Country

countries = struct('name', {'South Africa','Australia','Argentina','Chile'}, ...
                    'nMonths', {66, 143, 143, 142}, ...
                    'meanBottles', {meanBottlesSA, NaN, NaN, NaN});

for i = 2:numel(countries)
    cc = countries(i).name;
    mask = strcmp(string(Tv.wine_country), cc);
    countries(i).meanBottles = mean(Tv.numOfBottles(mask), 'omitnan');
end

fprintf('%-14s %10s %10s %14s %18s %18s\n', 'Country','N','MeanPrice','MeanBottles/Lot','OldTurnover($/mo)','CorrectedTurnover($/mo)');
for i = 1:numel(countries)
    cc = countries(i).name;
    row = avgTab(strcmp(avgTab.Country, cc), :);
    meanPrice = row.MeanPrice; N = row.N; nM = countries(i).nMonths;
    oldTurnover = (N/nM) * meanPrice;
    corrTurnover = oldTurnover * countries(i).meanBottles;
    fprintf('%-14s %10d %10.2f %14.3f %18.1f %18.1f\n', cc, N, meanPrice, countries(i).meanBottles, oldTurnover, corrTurnover);
end
