%% Slippage_PriceImpact_Proxy.m
% ---------------------------------------------------------------
% Referee 2, item 4: "stronger empirical evidence (not just
% implementation caps) that large trades in these thin markets
% wouldn't move prices/cause slippage."
%
% We do not have order-book or execution-sequence data, so this is a
% PROXY, not a direct measure of price impact: within each wine label
% x vintage x month cell that has 2+ lot sales, we regress each lot's
% log-price deviation from the cell's own mean log price on the log of
% that lot's size (number of bottles). A negative, significant slope
% means bigger lots tend to transact at a discount within the same
% wine/vintage/month -- evidence of price impact. A slope
% indistinguishable from zero means lot size is not systematically
% associated with price within a cell, in this sample.
%
% This is silent about impact ACROSS cells/time (e.g., a single very
% large lot dominating a whole month) and about buy-side price impact
% specifically -- it only speaks to within-month same-wine dispersion,
% which is what the lot-level data can actually support.
% ---------------------------------------------------------------

clear; clc;

% CURRENCY: without conversion, two lots in the same wine/vintage/month
% cell priced in different currencies would show a spurious price gap
% that is pure currency noise, not evidence of (or against) price impact.
% Dr. Verdickt confirmed (2026-11-02) this source file is uniformly
% AUD-denominated regardless of auction house or country, so every
% Verdickt-sourced row converts via normalize_lot_currency.m.

% WINDOW FIX (2026-11-12, audit-flagged): this script previously applied
% no date filter, so its reported Australia/Chile lot counts (76,674 /
% 496) included lots from Jan2012-Dec2012 and Dec2024, outside the
% Jan2013-Nov2024 window every other descriptive table (Table 1, Table 2 /
% AvgPrice_ByCountry.m) uses, producing a mismatch against those tables'
% 69,425 / 473. Restricted to the same window here.
DATE_START = datetime(2013,1,1);
DATE_END   = datetime(2024,11,30);

countries = struct( ...
    'name',   {'Argentina','Australia','Chile','South Africa'}, ...
    'source', {'verdickt','verdickt','verdickt','strauss'} );

results = table('Size',[0 7], ...
    'VariableTypes', ["string","double","double","double","double","double","double"], ...
    'VariableNames', ["Country","Lots","MultiLotCells","Slope","SE","tStat","R2"]);

for k = 1:numel(countries)
    cname = countries(k).name;

    if strcmp(countries(k).source, 'verdickt')
        T0 = readtable('Data Australia.xlsx','Sheet','Sheet1');
        T0 = T0(strcmp(T0.wine_country, cname), :);
        T0.ymDate = datetime(T0.ym, 'InputFormat', 'yyyy-MM');
        T0 = T0(T0.ymDate >= DATE_START & T0.ymDate <= DATE_END, :);
        T0 = normalize_lot_currency(T0);
        label = strtrim(string(T0.name));
        vint  = double(T0.vintage);
        ym    = string(T0.ym);
        price = double(T0.priceUSD);
        lotSize = double(T0.numOfBottles);
    else
        T0 = readtable('Strauss Wine All updated.xlsx','Sheet','SAOnly');
        label = strtrim(string(T0.WineName));
        vint  = double(T0.Vintage);
        if isdatetime(T0.AuctionDate)
            d = T0.AuctionDate;
        else
            d = datetime(T0.AuctionDate, 'ConvertFrom','excel');
        end
        ym    = string(d, 'yyyy-MM');
        price = double(T0.PricePer750ml);
        lotSize = double(T0.quantity);
    end

    valid = ~ismissing(label) & ~isnan(vint) & ~ismissing(ym) & ~isnan(price) & price > 0 ...
            & ~isnan(lotSize) & lotSize > 0;
    label = label(valid); vint = vint(valid); ym = ym(valid);
    logP  = log(price(valid)); logSize = log(lotSize(valid));

    cellID = label + "|" + string(vint) + "|" + ym;
    G = findgroups(cellID);
    nLots = numel(logP);

    cellMean = splitapply(@mean, logP, G);
    cellN    = splitapply(@numel, logP, G);
    dev      = logP - cellMean(G);              % within-cell log-price deviation

    keep = cellN(G) >= 2;                        % only cells with 2+ lots identify within-cell dispersion
    y = dev(keep); x = logSize(keep);
    % de-mean x within cell too, so the regression is purely within-cell
    xCellMean = splitapply(@mean, logSize(keep), findgroups(cellID(keep)));
    gk = findgroups(cellID(keep));
    xdev = x - xCellMean(gk);

    X = [ones(numel(xdev),1), xdev];
    b = X \ y;
    res = y - X*b;
    nObs = numel(y); Kp = 2;
    s2 = sum(res.^2) / (nObs - Kp);
    se = sqrt(diag(inv(full(X'*X))) * s2); %#ok<MINV>
    tstat = b(2) / se(2);
    r2 = 1 - sum(res.^2) / sum((y-mean(y)).^2);

    fprintf('%-14s: %d lots, %d multi-lot cells, %d within-cell obs | slope=%.4f (SE=%.4f, t=%.2f), R2=%.4f\n', ...
        cname, nLots, sum(cellN>=2), nObs, b(2), se(2), tstat, r2);

    results = [results; {cname, nLots, sum(cellN>=2), b(2), se(2), tstat, r2}]; %#ok<AGROW>
end

writetable(results, 'tab_slippage_proxy.csv');
disp(results);
