%% HybridSAWineIndex.m
% -------------------------------------------------------------------------
% Hybrid Hedonic–Repeat-Sales Index for South African Wine Prices
% Data: Strauss All.xlsx (Sheet1)
% Expected columns:
%   auction_house   – auction house name (ignored)
%   numOfBottles    – number of bottles sold in the lot
%   pricePerBottle  – price per bottle
%   ym              – 'YYYY-MM' string
%   wine_country    – country name (we’ll filter to 'South Africa')
%   name            – wine label (e.g. 'Meerlust Rubicon')
%   vintage         – numeric vintage year
% -------------------------------------------------------------------------

clearvars; close all; clc

% 1. Load & filter data
T = readtable('Strauss Wine All Updated.xlsx','Sheet','Top10');
% Keep only South African lots


% Compute log-price
 T.logP = log(T.Price_after_fees);

% T.logP = log(T.PricePer750ml);
%% 2. Create lot ID for repeat-sales: label + vintage
T.ID = strcat(T.WineName,'__',string(T.Vintage));

%% 3. Build hedonic design matrix
n = height(T);

% 3a. Time dummies from 'DateSold'
 timeCat    = categorical(T.DateSold);
% timeCat    = categorical(T.ActionDay);
timeLevels = categories(timeCat);
nT         = numel(timeLevels);
D_time     = dummyvar(timeCat);
D_time(:,1)= [];              % drop base level

% 3b. Label dummies from 'name'
nameCat    = categorical(T.WineName);
D_name     = dummyvar(nameCat);
D_name(:,1)= [];              % drop base

% 3c. Vintage dummies
vintCat    = categorical(string(T.Vintage));
D_vint     = dummyvar(vintCat);
D_vint(:,1)= [];              % drop base

% 3d. Assemble Xh = [1 | name | vintage | time]
Xh = [ ones(n,1), D_name, D_vint, D_time ];

%% 4. Identify repeat-sales pairs (exactly two sales per ID)
[G, uniqIDs]  = findgroups(T.ID);
grpCounts     = splitapply(@numel, T.ID, G);
repGroups     = find(grpCounts==2);
J             = numel(repGroups);

Yrs = nan(J,1);
Drs = nan(J, size(Xh,2));

for j = 1:J
    idx = find(G==repGroups(j));   % two rows
    i1 = idx(1);  i2 = idx(2);
    Yrs(j)   = T.logP(i2) - T.logP(i1);
    Drs(j,:) = Xh(i2,:)     - Xh(i1,:);
end

%% 5. Estimate error variances for GLS weights
% 5a. Hedonic OLS
b_h       = Xh \ T.logP;
res_h     = T.logP - Xh*b_h;
sigma2_e  = sum(res_h.^2) / (n - size(Xh,2));

% 5b. Repeat-sales OLS
b_rs      = Drs \ Yrs;
res_rs    = Yrs - Drs*b_rs;
sigma2_eta= sum(res_rs.^2) / (J - size(Drs,2));

%% 6. Stack and form GLS system
Yall = [ T.logP;       T.logP;      Yrs ];
Xall = [ Xh;           Xh;          Drs ];

w1   = 1/sigma2_e;
w2   = 1/(sigma2_e + sigma2_eta);
Wbig = [ w1*ones(n,1); w1*ones(n,1); w2*ones(J,1) ];

WX   = bsxfun(@times, Xall, Wbig);
WY   = Wbig .* Yall;
b_gls= (WX' * Xall) \ (WX' * Yall);

%% 7. Extract time-dummy coeffs → index
% time dummies start at column:
startIdx = 1 + size(D_name,2) + size(D_vint,2) + 1;
g = [ 0; b_gls(startIdx : startIdx + (nT-2)) ];  % base zero

% logIndex = g;
logIndex = smoothdata(g);
index    = exp(g);   % rebased to 1 at first period

%% 8. Plot the South African wine index by year-month
figure;
plot(1:nT, index, '-o','LineWidth',1.5);
grid on;
xticks(1:nT);
xticklabels(timeLevels);
xtickangle(45);
xlabel('Year–Month');
ylabel('Hybrid Index (Base = 1)');
title('Hybrid Hedonic–Repeat-Sales Index: South African Fine Wine');

%% 9. Save results
I = table(timeLevels, logIndex, index, ...
          'VariableNames', {'Period','LogIndex','IndexLevel'});
writetable(I,'HybridSAWineIndex.csv');

%% Display first 10 periods
disp(I(1:min(10,nT),:));

%% Interpolation

% Monthly dates from 01-Jun-2019 to 01-Nov-2024
startDate = datetime(2019,6,1);
endDate   = datetime(2024,11,1);

monthlyDates = (startDate : calmonths(1) : endDate).';   % column vector
% disp(monthlyDates)

SAIndex=interp1(timeLevels, index, monthlyDates, 'linear', 'extrap');

%%
II = table(monthlyDates, SAIndex, ...
          'VariableNames', {'Period','IndexLevel'});
writetable(II,'SAIndex.csv');