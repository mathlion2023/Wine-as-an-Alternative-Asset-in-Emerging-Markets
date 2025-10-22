%% RepeatSales_AustralianWineIndex_FIX.m
% Pure repeat-sales index with robust month mapping and diagnostics

clearvars; close all; clc;

FILE   = 'Data Australia.xlsx';
SHEET  = 'Sheet1';
COUNTRY= 'Argentina';  % <-- ensure correct filter
BASE_TO_100 = true;

%% 1) Load & filter
T = readtable(FILE,'Sheet',SHEET);

% Filter to Australia (fix the earlier 'Chile' typo)
isC = strcmpi(string(T.wine_country), COUNTRY);
T   = T(isC, :);

% Parse ym -> datetime (month start)
if ~isdatetime(T.ym)
    T.ym = datetime(string(T.ym),'InputFormat','yyyy-MM');
end
T.ym = dateshift(T.ym, 'start','month');

% Basic fields
T.pricePerBottle = double(T.pricePerBottle);
T.vintage        = double(T.vintage);
T.logP           = log(T.pricePerBottle);

% Item identity (label x vintage); adapt if you have a more granular ID
T.ID = strcat(string(T.name),'__',string(T.vintage));

% Sort by item then date
T = sortrows(T, {'ID','ym'});

%% 2) Stable month index using unique()
[uniqueMonths, ~, tIdx] = unique(T.ym);   % ascending, stable month index
nT = numel(uniqueMonths);

% quick sanity
fprintf('Distinct months in sample: %d (from %s to %s)\n', nT, ...
    datestr(uniqueMonths(1)), datestr(uniqueMonths(end)));

%% 3) Build REPEAT-SALES consecutive pairs
[G, ~] = findgroups(T.ID);
counts = splitapply(@numel, T.ID, G);
useG   = find(counts >= 2);

maxPairs = height(T) - numel(useG);  % rough upper bound
Yrs   = zeros(maxPairs, 1);
Dtime = zeros(maxPairs, nT-1);  % first month is base (dropped)
Wrs   = zeros(maxPairs, 1);
ptr   = 0;

for g = reshape(useG,1,[])
    idx = find(G == g);                  % rows for this item
    % consecutive pairs only
    for k = 1:(numel(idx) - 1)
        i1 = idx(k); i2 = idx(k+1);
        ptr = ptr + 1;

        % log price diff
        Yrs(ptr) = T.logP(i2) - T.logP(i1);

        % time-dummy diff (periods 2..nT)
        d = zeros(1, nT-1);
        j1 = tIdx(i1); j2 = tIdx(i2);   % month positions in 1..nT
        if j2 > 1, d(j2-1) = d(j2-1) + 1; end
        if j1 > 1, d(j1-1) = d(j1-1) - 1; end
        Dtime(ptr,:) = d;

        % holding-period weight (months)
        dM = calmonths(between(T.ym(i1), T.ym(i2), 'months'));
        if ~isfinite(dM) || dM <= 0, dM = 1; end
        Wrs(ptr) = 1 / dM;              % variance ∝ holding period
    end
end

% Trim to actual number of pairs
Yrs   = Yrs(1:ptr);
Dtime = Dtime(1:ptr, :);
Wrs   = Wrs(1:ptr);

fprintf('Repeat-sales items (>=2 trades): %d; pairs: %d\n', numel(useG), ptr);

% DIAGNOSTIC: if ptr==0 or all rows of Dtime are zeros
if ptr == 0
    error('No repeat-sales pairs found. Check the ID definition (label+vintage) or your filters.');
end
if all(all(Dtime == 0))
    error('Design matrix is all zeros. Month mapping failed. (This version fixes that via unique().)');
end

%% 4) (Optional) Outlier control (comment to skip)
z = (Yrs - median(Yrs)) / (1.4826*mad(Yrs,1));
keep = abs(z) <= 4;
Yrs = Yrs(keep); Dtime = Dtime(keep,:); Wrs = Wrs(keep);

%% 5) Weighted LS on RS system
Wsqrt = sqrt(Wrs);
Xw    = Dtime .* Wsqrt;
Yw    = Yrs   .* Wsqrt;

% Use pinv for numerical stability (rank-deficient months)
gamma_2_T = pinv(Xw) * Yw;   % (nT-1) x 1

% Rebuild log-index with base month = 0
logIndex = zeros(nT,1);
logIndex(2:end) = gamma_2_T;

% Convert to levels
IndexLevel = exp(logIndex);
IndexOut   = IndexLevel;
if BASE_TO_100, IndexOut = 100*IndexLevel; end

%% 6) Output and plot
outTbl = table(uniqueMonths, logIndex, IndexLevel, IndexOut, ...
    'VariableNames', {'Period','LogIndex','Index','IndexOut'});
writetable(outTbl, 'RepeatSales_ARG_Index.csv');

disp(outTbl(1:min(10,height(outTbl)),:));

figure('Color','w');
plot(outTbl.Period, outTbl.IndexOut, '-o','LineWidth',1.4); grid on;
xlabel('Month'); ylabel(sprintf('Index (Base = %d)', BASE_TO_100*100 + ~BASE_TO_100));
title(sprintf('Repeat-Sales Index: %s (Base = %d)', COUNTRY, BASE_TO_100*100 + ~BASE_TO_100));
xtickangle(45);

%% 7) (Optional) Interpolate onto a fixed monthly grid
startDate = outTbl.Period(1);
endDate   = outTbl.Period(end);
monthlyDates = (dateshift(startDate,'start','month') : calmonths(1) : dateshift(endDate,'start','month')).';
RS_interp = interp1(outTbl.Period, outTbl.IndexOut, monthlyDates, 'linear','extrap');
writetable(table(monthlyDates, RS_interp, 'VariableNames', {'Period','Index'}), ...
    'RepeatSales_ARG_Index_Interp.csv');
