%% Hedonic_Coefficients_Appendix.m
% ---------------------------------------------------------------
% Referee 2, item 8: report the underlying hedonic regression
% coefficients (vintage, producer/label, time effects) so readers can
% see how each attribute is priced across the four regional markets.
%
% This is a plain hedonic OLS (log price ~ label + vintage + time
% dummies), reported for its own sake -- distinct from the hybrid
% hedonic/repeat-sales GLS estimator in IndexConstruction*.m, which
% exists to build the price INDEX (time-dummy path only) rather than
% to report attribute prices.
%
% Label-level dummies are too numerous to report individually for
% every country (South Africa alone has ~2,300+ distinct labels), so
% this script reports:
%   - the FULL vintage-effect coefficient table per country (a
%     tractable number of vintages), and
%   - summary statistics for the label/producer fixed effects
%     (count, incremental R^2 from adding them, coefficient range),
%     rather than every individual label dummy.
% ---------------------------------------------------------------

clear; clc;

% CURRENCY: Verdickt-sourced rows (Argentina/Australia/Chile) convert via
% normalize_lot_currency.m -- Dr. Verdickt confirmed (2026-11-02) this
% source file is uniformly AUD-denominated regardless of auction house or
% country, superseding the earlier per-house EUR/GBP/ZAR inference.

countries = struct( ...
    'name',   {'Argentina','Australia','Chile','South Africa'}, ...
    'source', {'verdickt','verdickt','verdickt','strauss'} );

results = table('Size',[0 8], ...
    'VariableTypes', ["string","double","double","double","double","double","double","double"], ...
    'VariableNames', ["Country","Cells","Labels","Vintages","Periods", ...
                       "R2_TimeOnly","R2_TimeVintage","R2_Full"]);

for k = 1:numel(countries)
    cname = countries(k).name;

    if strcmp(countries(k).source, 'verdickt')
        T0 = readtable('Data Australia.xlsx','Sheet','Sheet1');
        T0 = T0(strcmp(T0.wine_country, cname), :);
        T0.ymDate = datetime(T0.ym, 'InputFormat', 'yyyy-MM');
        T0 = normalize_lot_currency(T0);
        label = strtrim(string(T0.name));
        vint  = double(T0.vintage);
        ym    = string(T0.ym);
        price = double(T0.priceUSD);
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
    end

    valid = ~ismissing(label) & ~isnan(vint) & ~ismissing(ym) & ~isnan(price) & price > 0;
    label = label(valid); vint = vint(valid); ym = ym(valid); price = price(valid);
    logP  = log(price);

    % Aggregate to one price per (label, vintage, ym) cell, matching
    % IndexConstruction.m's convention.
    [G0, aggLabel, aggVint, aggYm] = findgroups(label, vint, ym);
    aggLogP = splitapply(@mean, logP, G0);
    n = numel(aggLogP);

    labelCat = categorical(aggLabel);
    vintCat  = categorical(string(aggVint));
    timeCat  = categorical(aggYm);
    timeLevels = categories(timeCat);

    D_label = dummyvar(labelCat); D_label(:,1) = [];
    D_vint  = dummyvar(vintCat);  D_vint(:,1)  = [];
    D_time  = dummyvar(timeCat);  D_time(:,1)  = [];

    nLabel = size(D_label,2)+1; nVint = size(D_vint,2)+1; nTime = size(D_time,2)+1;

    % Nested models for incremental R^2: time-only -> +vintage -> +label
    Xt   = [ones(n,1), D_time];
    Xtv  = [ones(n,1), D_vint, D_time];
    Xtvl = [ones(n,1), D_label, D_vint, D_time];

    r2 = @(X) 1 - sum((aggLogP - X*(X\aggLogP)).^2) / sum((aggLogP-mean(aggLogP)).^2);
    R2_t   = r2(Xt);
    R2_tv  = r2(Xtv);
    R2_tvl = r2(Xtvl);

    % Vintage-effect table is reported from the time+vintage model (Xtv),
    % NOT the fully-saturated time+vintage+label model (Xtvl): once the
    % label dimension is a large fraction of the number of cells (as for
    % South Africa, 1223 labels over 3732 cells, mostly singleton/near-
    % singleton), Xtvl is rank-deficient and its coefficients/SEs are
    % unusable (confirmed: solving Xtvl for South Africa returns NaN
    % throughout). Xtv is always well-identified (vintage and time counts
    % are small relative to cells for every country here) and gives a
    % stable, comparable vintage-effect table across all four countries.
    bv   = Xtv \ aggLogP;
    resv = aggLogP - Xtv*bv;
    Kv   = size(Xtv,2);
    s2v  = sum(resv.^2) / (n-Kv);
    sev  = sqrt(diag(inv(full(Xtv'*Xtv))) * s2v); %#ok<MINV>

    vintIdx0 = 2:(1+size(D_vint,2));   % columns of Xtv holding vintage dummies
    vintLevels = categories(vintCat);
    vintCoef = [0; bv(vintIdx0)];
    vintSE   = [0; sev(vintIdx0)];
    vintT    = vintCoef ./ max(vintSE, eps);
    vintT(1) = NaN;

    % Label/producer effect: incremental R^2 and coefficient range from
    % the full model, only when that model is actually well-identified
    % (guards against reporting the South-Africa-style NaN degeneracy).
    b = Xtvl \ aggLogP;
    labelIdx0 = (2+size(D_vint,2)) : (1+size(D_vint,2)+size(D_label,2));
    labelCoef = b(labelIdx0);
    labelModelOK = ~any(isnan(b)) && ~isempty(labelCoef);
    if ~labelModelOK
        R2_tvl = NaN;
        labelCoef = [];
    end

    Tvint = table(string(vintLevels), vintCoef, vintSE, vintT, ...
        'VariableNames', {'Vintage','Coefficient','SE','tStat'});
    Tvint = sortrows(Tvint, 'Vintage');
    writetable(Tvint, sprintf('tab_hedonic_vintage_%s.csv', lower(strrep(cname,' ','_'))));

    fprintf('%-14s: %d cells, %d labels, %d vintages, %d periods | R2 time=%.3f, +vintage=%.3f, +label=%.3f\n', ...
        cname, n, nLabel, nVint, nTime, R2_t, R2_tv, R2_tvl);

    results = [results; {cname, n, nLabel, nVint, nTime, R2_t, R2_tv, R2_tvl}]; %#ok<AGROW>

    labelLevels = categories(labelCat);
    if numel(labelLevels) <= 1
        fprintf('  Only 1 distinct label ("%s") -- no label-effect coefficients to report.\n', labelLevels{1});
    elseif ~labelModelOK
        fprintf('  Label dimension (%d labels over %d cells) is too sparse for a well-identified label effect -- not reported.\n', ...
            numel(labelLevels), n);
    else
        fprintf('  Label-effect coefficient range: [%.3f, %.3f] (base label = "%s")\n', ...
            min(labelCoef), max(labelCoef), labelLevels{1});
    end
end

writetable(results, 'tab_hedonic_summary.csv');
disp(results);
