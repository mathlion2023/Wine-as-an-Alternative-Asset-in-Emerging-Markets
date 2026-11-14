%% SHORT_Window_Interp_Check.m (2026-11-12)
% Robert/audit comment Q6: Online Appendix Table S2's note cites Argentina
% 44% / Chile 13% interpolated "in this window" (the SHORT panel window,
% 2019:06-2024:11, 66 months), but those two figures are actually the
% NATIVE-window (2013:01/02-2024:11, 143/142 months) shares from main
% Table 1. This script computes the SHORT-window-specific zero-trade share
% directly, read-only, reusing Sparse_Month_Disclosure.m's loading logic.
clear; clc;

shortStart = datetime(2019,6,1);
shortEnd   = datetime(2024,11,1);

countries = struct( ...
    'key',   {'ARGFW', 'CHIFW'}, ...
    'filter',{'Argentina', 'Chile'});

Tver = readtable('Data Australia.xlsx','Sheet','Sheet1');
Tver.ymDate = datetime(Tver.ym, 'InputFormat', 'yyyy-MM');
Tver = normalize_lot_currency(Tver);

fullGrid = (shortStart:calmonths(1):shortEnd)';
nTotalMonths = numel(fullGrid);

for c = 1:numel(countries)
    key = countries(c).key;
    Tc = Tver(strcmp(Tver.wine_country, countries(c).filter), :);
    Tc = Tc(Tc.ymDate >= shortStart & Tc.ymDate <= shortEnd, :);

    good = isfinite(Tc.priceUSD) & Tc.priceUSD > 0 & ~isnat(Tc.ymDate) & isfinite(Tc.vintage);
    Tc = Tc(good,:);

    monthOf = dateshift(Tc.ymDate, 'start', 'month');
    uMonths = unique(monthOf);
    uMonthsStr = cellstr(datestr(uMonths,'yyyy-mm'));

    nZero = 0; nTraded = 0;
    for m = 1:nTotalMonths
        kstr = datestr(fullGrid(m),'yyyy-mm');
        if any(strcmp(uMonthsStr, kstr))
            nTraded = nTraded + 1;
        else
            nZero = nZero + 1;
        end
    end

    pct = 100*nZero/nTotalMonths;
    fprintf('%s SHORT window (%s to %s): %d total months | %d traded | %d zero-trade (interpolated) | %.1f%%\n', ...
        key, datestr(shortStart,'yyyy-mm'), datestr(shortEnd,'yyyy-mm'), nTotalMonths, nTraded, nZero, pct);
end
