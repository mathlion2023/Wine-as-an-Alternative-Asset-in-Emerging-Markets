%% RepeatSales_SAWineIndex_AGG.m
% Pure repeat-sales index for South African wine with monthly aggregation
% (robust to string months like '01-May-2022' and duplicate trades per month)

clear; clc;

FILE  = 'Strauss Wine All Updated.xlsx';
SHEET = 'Top10';
BASE_TO_100 = true;   % base month = 100
RIDGE = 1e-8;         % tiny ridge for numerical stability

%% 1) Load and map columns
T = readtable(FILE, 'Sheet', SHEET);

% Required columns (edit names here if different)
req = {'WineName','Vintage','DateSold'};
assert(all(ismember(req, T.Properties.VariableNames)), ...
    'Missing columns: %s', strjoin(setdiff(req, T.Properties.VariableNames), ', '));

% Price column: prefer per-750ml if available
priceCol = 'Price_after_fees';
if ismember('PricePer750ml', T.Properties.VariableNames)
    priceCol = 'PricePer750ml';
end
T.Price   = double(T.(priceCol));
T.Vintage = double(T.Vintage);

% Parse DateSold → month start; handle strings like '01-May-2022'
if ~isdatetime(T.DateSold)
    % Try dd-MMM-yyyy (matches '01-May-2022'), then fallback
    try
        T.DateSold = datetime(string(T.DateSold), 'InputFormat','dd-MMM-yyyy');
    catch
        try
            T.DateSold = datetime(string(T.DateSold), 'InputFormat','yyyy-MM-dd');
        catch
            T.DateSold = datetime(string(T.DateSold)); % let MATLAB infer
        end
    end
end
T.Month = dateshift(T.DateSold, 'start','month');

% Basic filters
T = T(isfinite(T.Price) & T.Price>0 & ~ismissing(T.WineName) & ~isnan(T.Vintage), :);

% ID definition (extend with format/bottle size if needed)
Name = lower(strtrim(string(T.WineName)));
T.ID = Name + "__" + string(T.Vintage);

%% 2) Monthly aggregation: one obs per (ID, Month)
G = findgroups(T.ID, T.Month);
Agg = table;
Agg.ID    = splitapply(@(x) x(1), T.ID,    G);
Agg.Month = splitapply(@(x) x(1), T.Month, G);
Agg.Price = splitapply(@median,    T.Price, G);   % median per (ID, Month)
Agg.logP  = log(Agg.Price);

% Sort
Agg = sortrows(Agg, {'ID','Month'});

fprintf('After monthly aggregation: %d rows, %d unique IDs.\n', ...
    height(Agg), numel(unique(Agg.ID)));

%% 3) Build consecutive repeat-sales pairs from aggregated data
[GI, ~] = findgroups(Agg.ID);
counts  = splitapply(@numel, Agg.ID, GI);
hasPairs= find(counts >= 2);

pairs = [];
for g = reshape(hasPairs,1,[])
    idx = find(GI == g);
    % consecutive pairs only (1→2, 2→3, ...)
    pairs = [pairs; [idx(1:end-1), idx(2:end)]]; %#ok<AGROW>
end
if isempty(pairs), error('No repeat-sales pairs after aggregation.'); end

% Drop same-month pairs (should be none after aggregation, but just in case)
sameMonth = Agg.Month(pairs(:,1)) == Agg.Month(pairs(:,2));
pairs(sameMonth, :) = [];

fprintf('Consecutive RS pairs (post-drop same-month): %d\n', size(pairs,1));

% Construct Y and (t1,t2)
Yrs   = Agg.logP(pairs(:,2)) - Agg.logP(pairs(:,1));
t1    = Agg.Month(pairs(:,1));
t2    = Agg.Month(pairs(:,2));

%% 4) Month basis from sorted unique months appearing IN PAIRS
monthsUsed = unique([t1; t2], 'sorted');
nT = numel(monthsUsed);
fprintf('RS-identifying months: %d (from %s to %s)\n', nT, ...
    datestr(monthsUsed(1)), datestr(monthsUsed(end)));

% Map each pair’s t1,t2 to 1..nT
[~, j1] = ismember(t1, monthsUsed);
[~, j2] = ismember(t2, monthsUsed);

% Design matrix for time-dummy differences with month 1 as base
Dtime = zeros(size(pairs,1), nT-1);  % columns = months 2..nT
for r = 1:size(pairs,1)
    if j2(r) > 1, Dtime(r, j2(r)-1) = Dtime(r, j2(r)-1) + 1; end
    if j1(r) > 1, Dtime(r, j1(r)-1) = Dtime(r, j1(r)-1) - 1; end
end
if all(all(Dtime==0)), error('Design matrix is all zeros — check date parsing.'); end

% Weights: inverse holding-period months
dM = calmonths(between(t1, t2, 'months'));
dM(~isfinite(dM) | dM<=0) = 1;
Wrs = 1 ./ dM;

%% 5) Weighted LS (with tiny ridge for stability)
Wsqrt = sqrt(Wrs);
Xw = Dtime .* Wsqrt;
Yw = Yrs   .* Wsqrt;

XtX = Xw.' * Xw;
XtY = Xw.' * Yw;
gamma_2_T = (XtX + RIDGE*eye(size(XtX))) \ XtY;   % (nT-1) x 1

% Rebuild log-index with base month = 0
logIndex = zeros(nT,1);
logIndex(2:end) = gamma_2_T;

IndexLevel = exp(logIndex);
IndexOut   = IndexLevel;
if BASE_TO_100, IndexOut = 100 * IndexLevel; end

%% 6) Output + diagnostics
outTbl = table(monthsUsed, logIndex, IndexLevel, IndexOut, ...
    'VariableNames', {'Period','LogIndex','Index','IndexOut'});
writetable(outTbl, 'RepeatSales_SA_Index.csv');

% Show first 12 rows to verify nonzero path
disp(outTbl(1:min(12,height(outTbl)),:));

% Print a few pairs to sanity-check
fprintf('\nSample pairs (first 10):\n');
K = min(10, size(pairs,1));
for k = 1:K
    fprintf('%s -> %s | ΔM=%2d | ΔlogP=% .4f\n', ...
        datestr(t1(k),'yyyy-mm'), datestr(t2(k),'yyyy-mm'), dM(k), Yrs(k));
end

% Rank check
rnk = rank(Dtime);
if rnk < size(Dtime,2)
    warning('Rank(Dtime)=%d < %d columns. Some months weakly identified; ridge helps.', rnk, size(Dtime,2));
end

%% 7) Plot
figure('Color','w');
plot(outTbl.Period, outTbl.IndexOut, '-o','LineWidth',1.5);
grid on; xlabel('Month');
ylabel(sprintf('Index (Base = %d)', BASE_TO_100*100 + ~BASE_TO_100));
title('Repeat-Sales Price Index: South Africa (Aggregated, Base = 100)');
xtickangle(45);

%% 8) Optional: full monthly grid interpolation (consistent axis)
startDate = dateshift(outTbl.Period(1), 'start','month');
endDate   = dateshift(outTbl.Period(end), 'start','month');
gridMonths = (startDate : calmonths(1) : endDate).';
RS_grid    = interp1(outTbl.Period, outTbl.IndexOut, gridMonths, 'linear','extrap');
writetable(table(gridMonths, RS_grid, 'VariableNames', {'Period','Index'}), ...
           'RepeatSales_SA_Index_Interp.csv');
