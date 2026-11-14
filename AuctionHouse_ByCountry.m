%% AuctionHouse_ByCountry.m (2026-11-12)
% Robert/audit comment Q3: Data section describes the 45-house auction_house
% field but does not enumerate sources by country. This script produces a
% read-only per-country house count/share breakdown directly from
% Data Australia.xlsx, reusing the same loading/currency logic as
% Build_RS_Hybrid_AllCountries.m (no re-estimation, counting only).
clear; clc;

Tver = readtable('Data Australia.xlsx','Sheet','Sheet1');
Tver.ymDate = datetime(Tver.ym, 'InputFormat', 'yyyy-MM');
Tver = normalize_lot_currency(Tver);

countries = {'Australia','Argentina','Chile'};
for c = 1:numel(countries)
    Tc = Tver(strcmp(Tver.wine_country, countries{c}), :);
    good = isfinite(Tc.priceUSD) & Tc.priceUSD > 0;
    Tc = Tc(good,:);
    n = height(Tc);
    [houses, ~, ig] = unique(string(Tc.auction_house));
    cnts = accumarray(ig,1);
    [cnts, ord] = sort(cnts,'descend');
    houses = houses(ord);
    pct = 100*cnts/n;
    fprintf('\n=== %s (n=%d lots, %d distinct houses) ===\n', countries{c}, n, numel(houses));
    nShow = min(8, numel(houses));
    for h = 1:nShow
        fprintf('  %-28s %6d lots (%.1f%%)\n', houses(h), cnts(h), pct(h));
    end
    if numel(houses) > nShow
        fprintf('  (remaining %d houses, %.1f%% of lots)\n', numel(houses)-nShow, sum(pct(nShow+1:end)));
    end
end
