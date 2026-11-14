%% Inspect_Currency_Fields.m (2026-10-29, Faff currency-verification request)
% Diagnostic, read-only: inspect Data Australia.xlsx's actual columns for
% any sale-location/currency hint beyond auction_house, and pull out the
% specific rows Robert flagged (Acker, Winefield's) for manual inspection.
clear; clc;
Tver = readtable('Data Australia.xlsx','Sheet','Sheet1');
fprintf('Columns in Data Australia.xlsx:\n');
disp(Tver.Properties.VariableNames');

house = string(Tver.auction_house);
u = unique(house);
fprintf('\n%d unique auction_house values.\n', numel(u));

% Residual (no-FX) houses currently swept into isUSD by elimination
residualHouses = ["Bruun Rasmussen","Bukowskis","Baghera","Steinfels","Waddingtons","Webb's","Webbs","Winefields","Winefield's","Winefield"];
fprintf('\n--- Residual-house row counts by country ---\n');
for i = 1:numel(residualHouses)
    mask = contains(house, residualHouses(i), 'IgnoreCase', true);
    if any(mask)
        tbl = Tver(mask,:);
        ymc = sort(string(tbl.ym));
        fprintf('%s : n=%d, countries=%s, date range %s to %s\n', residualHouses(i), sum(mask), ...
            strjoin(unique(string(tbl.wine_country)),','), ...
            ymc(1), ymc(end));
    end
end

fprintf('\n--- Acker rows: date spread (looking for any HK-sale clue) ---\n');
isAcker = contains(house, "Acker", 'IgnoreCase', true);
fprintf('Acker n=%d, countries=%s\n', sum(isAcker), strjoin(unique(string(Tver.wine_country(isAcker))),','));
disp(unique(house(isAcker)));

fprintf('\n--- Winefield rows: all fields for inspection ---\n');
isWF = contains(house, "Winefield", 'IgnoreCase', true) | contains(house, "Webb", 'IgnoreCase', true);
fprintf('Winefield/Webbs n=%d\n', sum(isWF));
if any(isWF)
    disp(Tver(isWF, :));
end

fprintf('\n--- Full per-house row counts (for the record) ---\n');
T2 = table(u, 'VariableNames', {'house'});
cnt = zeros(numel(u),1);
for i=1:numel(u)
    cnt(i) = sum(house==u(i));
end
T2.n = cnt;
T2 = sortrows(T2, 'n', 'descend');
disp(T2);
