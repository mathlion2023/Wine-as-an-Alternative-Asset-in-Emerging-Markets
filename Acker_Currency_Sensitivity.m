%% Acker_Currency_Sensitivity.m -- SUPERSEDED/OBSOLETE (2026-11-02)
% This script quantified how much it could matter that Acker Merrall &
% Condit's lots had no recoverable sale-location field, under the
% then-live per-house currency inference (Acker held at a documented USD
% prior). Dr. Verdickt has since confirmed directly that every price in
% Data Australia.xlsx, including Acker's, is AUD -- there is no remaining
% currency ambiguity for this house to quantify. Retained for the
% project's own record, not run as part of the pipeline and not cited in
% the manuscript.
clear; clc;
startDate = datetime(2013,1,1); endDate = datetime(2024,11,1);
load('MasterData_USD_2013_2024.mat', 'M');
grid = M.Period;

Tver = readtable('Data Australia.xlsx','Sheet','Sheet1');
Tver.ymDate = datetime(Tver.ym, 'InputFormat', 'yyyy-MM');
house = string(Tver.auction_house);
isAcker = startsWith(house, "Acker", 'IgnoreCase', true);
fprintf('Acker rows: Australia=%d, Argentina=%d, Chile=%d\n', ...
    sum(isAcker & Tver.wine_country=="Australia"), sum(isAcker & Tver.wine_country=="Argentina"), sum(isAcker & Tver.wine_country=="Chile"));

Tver = normalize_lot_currency(Tver);

countries = struct('key',{'AUSFW','ARGFW','CHIFW'},'filter',{'Australia','Argentina','Chile'});
for c = 1:numel(countries)
    key = countries(c).key; filt = countries(c).filter;
    Tc = Tver(strcmp(Tver.wine_country, filt) & Tver.ymDate>=startDate & Tver.ymDate<=endDate, :);
    good = isfinite(Tc.priceUSD) & Tc.priceUSD>0 & ~isnat(Tc.ymDate) & isfinite(Tc.vintage);
    Tc = Tc(good,:);

    % Baseline (Acker included, as in the published pipeline)
    [Rbase] = fogartyJonesHybridRS(string(Tc.name), Tc.vintage, Tc.ymDate, log(Tc.priceUSD), ~strcmp(key,'SAFW'));
    Rbase.Period = dateshift(Rbase.Period,'start','month');

    % Alternative: Acker rows dropped entirely
    isAckerC = startsWith(string(Tc.auction_house), "Acker", 'IgnoreCase', true);
    Tx = Tc(~isAckerC, :);
    nDropped = sum(isAckerC);
    [Ralt] = fogartyJonesHybridRS(string(Tx.name), Tx.vintage, Tx.ymDate, log(Tx.priceUSD), ~strcmp(key,'SAFW'));
    Ralt.Period = dateshift(Ralt.Period,'start','month');

    logB = interp1(datenum(Rbase.Period), Rbase.LogIndex_HY, datenum(grid), 'linear');
    logA = interp1(datenum(Ralt.Period), Ralt.LogIndex_HY, datenum(grid), 'linear');
    both = ~isnan(logB) & ~isnan(logA);
    dB = diff(logB(both)); dA = diff(logA(both));
    corrRet = corr(dB(:), dA(:), 'rows','complete');
    volB = std(dB,'omitnan'); volA = std(dA,'omitnan');
    rmseLevel = sqrt(mean((exp(logB(both))-exp(logA(both))).^2));
    fprintf(['\n%s: dropping %d Acker rows (of %d, %.1f%%) -> return correlation with baseline = %.3f, ', ...
        'monthly return vol %.3f (baseline) vs %.3f (ex-Acker), level RMSE = %.2f index points.\n'], ...
        key, nDropped, height(Tc), 100*nDropped/height(Tc), corrRet, volB, volA, rmseLevel);
end
