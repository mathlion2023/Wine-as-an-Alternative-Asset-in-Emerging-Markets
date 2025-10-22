%% HybridAustralianWineIndex.m
% -------------------------------------------------------------------------
% Hybrid Hedonic–Repeat-Sales Index for Australian Wine Prices
% Data: Data Australia.xlsx (Sheet1)
% Columns expected:
%   auction_house   – name of auction house (ignored)
%   numOfBottles    – number of bottles sold in the lot
%   pricePerBottle  – price (USD) per bottle
%   ym              – year-month string, e.g. '2012-01'
%   wine_country    – country name (we’ll filter to Australia)
%   name            – wine label (e.g. 'Penfolds Bin 707')
%   vintage         – numeric vintage year
% -------------------------------------------------------------------------

% 0. Clean up
clearvars; close all; clc

% 1. Load & filter data
T = readtable('Data Australia.xlsx','Sheet','Sheet1');

% Keep only Australian lots
isAU = strcmp(T.wine_country,'Chile');
T = T(isAU,:);

% Log-price per bottle
T.logP = log(T.pricePerBottle);

%% 2. Build “ID” for repeat-sales: same label & vintage
T.ID = strcat(T.name,'__',string(T.vintage));

%% 3. Build hedonic design matrix Xh
n = height(T);

% 3a. Time dummies from ym
timeCat   = categorical(T.ym);
timeLevels= categories(timeCat);
nT        = numel(timeLevels);
D_time    = dummyvar(timeCat);
D_time(:,1)=[];           % drop first level as base
% 3b. Label dummies from name (only include labels with ≥2 sales)
nameCat   = categorical(T.name);
lblCounts = countcats(nameCat);
keepLbls  = nameCat; 
% (option) you could group rare labels into “Other” – here we include all
D_name    = dummyvar(nameCat);
D_name(:,1)=[];           % base label dropped
% 3c. Vintage dummies
vintCat   = categorical(string(T.vintage));
vintLevels= categories(vintCat);
D_vint    = dummyvar(vintCat);
D_vint(:,1)=[];           % drop first vintage

% 3d. Assemble Xh: [intercept | name dummies | vintage dummies | time dummies]
Xh = [ ones(n,1), D_name, D_vint, D_time ];

%% 4. Identify repeat-sales pairs (exactly 2 sales per ID)
[G, uniqIDs]    = findgroups(T.ID);
countsByGroup   = splitapply(@numel, T.ID, G);
repGroupIDs     = find(countsByGroup==2);
J               = numel(repGroupIDs);

Yrs = nan(J,1);
Drs = nan(J, size(Xh,2));

for j=1:J
    rows = find(G==repGroupIDs(j));    % two rows
    i1 = rows(1);  i2 = rows(2);
    Yrs(j)     = T.logP(i2) - T.logP(i1);
    Drs(j,:)   = Xh(i2,:)     - Xh(i1,:);
end

%% 5. Estimate error variances for GLS weights
% 5a. Hedonic OLS
b_h    = Xh \ T.logP;
res_h  = T.logP - Xh*b_h;
sigma2_eps = sum(res_h.^2) / (n - size(Xh,2));

% 5b. Repeat-sales OLS
b_rs     = Drs \ Yrs;
res_rs   = Yrs - Drs*b_rs;
sigma2_eta = sum(res_rs.^2) / (J - size(Drs,2));

%% 6. Stack for GLS
Yall = [ T.logP; T.logP; Yrs ];
Xall = [ Xh;       Xh;       Drs ];

w1    = 1/sigma2_eps;
w2    = 1/(sigma2_eps + sigma2_eta);
Wbig  = [ w1*ones(n,1); w1*ones(n,1); w2*ones(J,1) ];

% GLS solution: (X' W X) β = X' W Y
WX    = bsxfun(@times, Xall, Wbig);
WY    = Wbig .* Yall;
b_gls = (WX' * Xall) \ (WX' * Yall);

%% 7. Extract time-dummy coefficients → index
% time dummies occupy columns: 1 + (size(D_name,2)) + (size(D_vint,2)) + [1:(nT-1)]
startIdx = 1 + size(D_name,2) + size(D_vint,2) + 1;
g = [ 0; b_gls(startIdx : startIdx + (nT-2)) ];   % zero for base period
logIndex = g;
index    = exp(logIndex);  % base = 1

%% 8. Plot the hybrid index
yearLabels = cellfun(@(s) s(1:4), timeLevels, 'UniformOutput', false);

figure;
plot(1:nT, index, '-o', 'LineWidth', 1.4);
grid on;
xticks(1:nT);
xticklabels(yearLabels);
xtickangle(45);
xlabel('Year');
ylabel('Price Index (Base = 1)');
title('Hybrid Hedonic–Repeat-Sales Index: Australian Fine Wine');

%% 9. (Optional) Save
I = table(timeLevels, logIndex, index, ...
          'VariableNames', {'Period','LogIndex','IndexLevel'});
writetable(I,'HybridAUWineIndex.csv');

%% 10. Display first few values
disp(I(1:10,:));


%% Interpolation

% Monthly dates from 01-Jun-2019 to 01-Nov-2024
startDate = datetime(2019,6,1);
endDate   = datetime(2024,11,1);

monthlyDates = (startDate : calmonths(1) : endDate).';   % column vector
% disp(monthlyDates)
monthlyDates.Format = 'yyyy-MM';
SAIndex=interp1(timeLevels, index, monthlyDates, 'linear', 'extrap');

%%
II = table(monthlyDates, SAIndex, ...
          'VariableNames', {'Period','IndexLevel'});
writetable(II,'SAIndex.csv');