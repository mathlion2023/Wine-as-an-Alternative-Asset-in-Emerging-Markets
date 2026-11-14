%% Build_Extra_FX.m -- SUPERSEDED/OBSOLETE (2026-11-02)
% Built month-end USD-per-unit rates for DKK, SEK, CHF, CAD, NZD, SGD to
% support the per-house currency inference used before Dr. Verdickt
% confirmed directly that Data Australia.xlsx is uniformly AUD-denominated.
% No script in the pipeline consumes FX_Extra_Currencies.csv anymore
% (normalize_lot_currency.m converts every row via AUDUSD alone). Retained
% for the project's own record; not run as part of the pipeline.
%
% Original description, for reference: builds month-end USD-per-unit rates
% for DKK, SEK, CHF, CAD, NZD from Federal Reserve H.10 daily series
% (fetched from FRED's public CSV endpoint: DEXDNUS, DEXSDUS, DEXSZUS,
% DEXCAUS, DEXUSNZ), to replace the face-value-as-USD treatment of Bruun
% Rasmussen (DKK), Bukowskis (SEK), Baghera/Steinfels (CHF), Waddingtons
% (CAD), Webb's/Winefields (NZD). DEXDNUS/DEXSDUS/DEXSZUS/DEXCAUS quote
% local units per USD -> invert for USD-per-unit; DEXUSNZ already quotes
% USD per NZD.
clear; clc;
series = struct('id',{'DEXDNUS','DEXSDUS','DEXSZUS','DEXCAUS','DEXUSNZ','DEXSIUS'}, ...
                 'name',{'DKKUSD','SEKUSD','CHFUSD','CADUSD','NZDUSD','SGDUSD'}, ...
                 'invert',{true,true,true,true,false,true});

grid = (datetime(2013,1,1):calmonths(1):datetime(2024,12,1))';
out = table(grid, 'VariableNames', {'Period'});

for s = 1:numel(series)
    raw = readtable(fullfile(tempdir,'wine_fx',[series(s).id '.csv']));
    raw.Properties.VariableNames = {'Date','Rate'};
    raw.Date = datetime(raw.Date);
    raw = raw(~isnan(raw.Rate), :);
    monthKey = dateshift(raw.Date, 'start', 'month');
    vals = nan(numel(grid),1);
    for m = 1:numel(grid)
        idx = find(monthKey == grid(m));
        if ~isempty(idx)
            vals(m) = raw.Rate(idx(end)); % last available trading day of month (matches Bloomberg PX_LAST month-end convention used elsewhere in this pipeline)
        end
    end
    if series(s).invert
        vals = 1 ./ vals;
    end
    nMissing = sum(isnan(vals));
    if nMissing > 0
        vals = fillmissing(vals, 'previous');
        vals = fillmissing(vals, 'next');
        fprintf('%s: %d month(s) had no FRED trading day, filled via carry-forward/back.\n', series(s).name, nMissing);
    end
    out.(series(s).name) = vals;
end

writetable(out, 'FX_Extra_Currencies.csv');
fprintf('Saved FX_Extra_Currencies.csv (%d months, %s to %s), source: Federal Reserve H.10 (FRED DEXDNUS/DEXSDUS/DEXSZUS/DEXCAUS/DEXUSNZ).\n', ...
    height(out), string(out.Period(1)), string(out.Period(end)));
disp(out(1:3,:));
