%% Cumulative returns for fixed-weight portfolios (2013–2024)
% Assumptions:
% - Monthly returns for 9 assets in columns, aligned with 'assetNames' below
% - Full monthly rebalancing to the fixed weights
% - If you have log-returns, set isLogReturn = true
% - Date vector covers 2019-01 ... 2024-12 (or your exact sample)

clear; clc;

%% ====== INPUTS ======

% 1) Provide your T x 9 monthly returns matrix R and a T x 1 datetime vector 'dates'.
%    Replace this block with your own loader.
% Example loader (CSV with header row matching 'assetNames'):
% T = readtable('monthly_returns.csv'); 
% dates = datetime(T.Date); T.Date = [];
% R = table2array(T);  % simple returns in decimal (e.g. 0.02 = 2%)
% Repointed (2026-08-25) at the USD master dataset (Build_MasterData_USD.m
% output); re-repointed (2026-10-08) at the SHORT/baseline 4-country panel
% (Jun2019-Nov2024) -- this 8-asset system needs South Africa, which has
% no real data before Jun 2019 (see portfolio_paths_tc.m header for detail).
% Fixed weight vectors below (W.GMV etc.) were optimized for the original
% 9-asset universe in the manuscript and are NOT re-derived here. "Bond"
% dropped (2026-08-25, per user instruction) -- its GlobalAgg_USD mapping
% was an unconfirmed assumption; see below for how the weights were
% adjusted to compensate.
load('MasterData_USD_2019_2024_4country.mat', 'M');
srcCols = ["SA_wine_USD","AUS_wine_USD","ARG_wine_USD","CHL_wine_USD","SPX_USD","MSCIEM_USD","Gold_USD","LivEx100_USD"];
data = M{:, srcCols};
date = datenum(M.Period);

% Data=table2array(data);
R   = diff(log(data)); % T×N returns

% For illustration only (remove in production):
% error('Replace the demo loader with your own data loader that sets R (Tx9) and dates (Tx1 datetime).');

% 2) Are these log-returns?
isLogReturn = true;  % set true if R contains ln(1+r)

% 3) Asset order must match your table:
assetNames = ["SAfw","Ausfw","Argfw","Chifw","SP500","MSCIEM","Gold","Liv_ex100"];

% 4) Fixed weights (percent -> decimal) from your table.
% Order: SAfw, Ausfw, Argfw, Chifw, SP500, MSCIEM, Bond, Gold, Liv_ex100
W9 = struct();
W9.GMV          = [ 1  ,  4,  2,  3, 11, 11, 22, 19, 26 ]'/100;
W9.Tangency     = [ 1  ,  4,  2,  2, 20, 15, 10, 27, 21 ]'/100;
W9.RiskParity   = [ 1  ,  1,  1,  1, 44,  1, 30, 19,  1 ]'/100;
W9.CVaR         = [ 1  ,  1,  1,  0,  9,  0,  9, 26, 54 ]'/100;
W9.Turnoverpen  = [ 1  ,  4,  2,  2, 19, 15, 11, 26, 20 ]'/100;
W9.BL_Tangency  = [ 0  ,  2,  2,  0, 32, 19,  0, 37,  7 ]'/100;

% Bond (position 7) dropped and each vector rescaled to sum back to 1
% across the remaining 8 assets -- this preserves the *relative* weights
% among the other assets from the original table; it is NOT a
% re-optimization for the new 8-asset universe.
bondIdx = 7;
namesW9 = fieldnames(W9);
W = struct();
for k = 1:numel(namesW9)
    w = W9.(namesW9{k});
    w(bondIdx) = [];
    W.(namesW9{k}) = w / sum(w);
end

% Optional: sanity checks
% namesW = fieldnames(W);
% for k = 1:numel(namesW)
%     s = namesW{k};
%     sum(W.(s))
%     if abs(sum(W.(s)) - 1) ~= 0
%         error('Weights for %s do not sum to 1.', s);
%     end
%     if any(W.(s) < -1e-12)
%         error('Negative weight found in %s (expected long-only).', s);
%     end
% end


[T, N] = size(R);
assert(N == numel(assetNames), 'R must have 8 columns matching assetNames.');

% Handle any NaNs by row-wise deletion (or choose smarter imputation if preferred)
good = all(isfinite(R), 2);
R = R(good, :);
date = date(good);

%% ====== HELPER: compute portfolio series ======
% Given returns matrix R (simple or log) and weight vector w (N x 1)
% return a structure with monthly portfolio returns and cumulative index (base=100).

computePortfolio = @(R,w,isLog) struct( ...
    'r',   R * w, ...
    'idx', localCumIndex(R*w, isLog) ...
);

%% ====== BUILD ALL PORTFOLIOS ======
portNames = fieldnames(W);
PORT = struct();
for k = 1:numel(portNames)
    s = portNames{k};
    PORT.(s) = computePortfolio(R, W.(s), isLogReturn);
end

%% ====== SUMMARY TABLE: final cumulative return and annualized stats ======
% Annualization assumes monthly data.
toAnnual = 12;

summ = table('Size',[numel(portNames) 5], ...
             'VariableTypes',["string","double","double","double","double"], ...
             'VariableNames',["Portfolio","CumRet","AnnMean","AnnVol","Sharpe"]);
for k = 1:numel(portNames)
    s = portNames{k};
    r = PORT.(s).r;           % monthly portfolio returns (simple or log)
    if isLogReturn
        % Convert log returns to simple for statistics
        r_s = exp(r) - 1;
    else
        r_s = r;
    end
    % Cumulative return over sample
    if isLogReturn
        cumRet = exp(sum(r)) - 1;
    else
        cumRet = prod(1 + r) - 1;
    end
    % Annualized stats
    m  = mean(r_s,'omitnan');
    sd = std(r_s,0,'omitnan');
    annMean = (1 + m)^toAnnual - 1;
    annVol  = sd * sqrt(toAnnual);
    sharpe  = (annMean) / max(annVol,eps);   % r_f = 0

    summ.Portfolio(k) = string(s);
    summ.CumRet(k)    = cumRet;
    summ.AnnMean(k)   = annMean;
    summ.AnnVol(k)    = annVol;
    summ.Sharpe(k)    = sharpe;
end

disp('==== Cumulative return (2013–2024) and annualized stats ====');
disp(summ);

%% ====== PLOTS ======

% 1) Cumulative indices (base = 100)
figure('Color','w'); hold on; box on;
for k = 1:numel(portNames)
    s = portNames{k};
    plot(date, PORT.(s).idx, 'LineWidth', 1.6, 'DisplayName', strrep(s,'_','\_'));
end
%yline(100,'k:');
xlabel('Date'); ylabel('Index (base = 100)');
% title('Cumulative Performance of Fixed-Weight Portfolios (2013–2024)');
legend('Location', 'best'); grid on;
datetick
axis tight
% 2) Bar chart of total cumulative return
figure('Color','w');
bar(categorical(summ.Portfolio), summ.CumRet);
ylabel('Cumulative Return (decimal)');
title('Cumulative Return 2019–2024');
grid on;

%% ====== export summary ======
writetable(summ, 'portfolio_cum_returns_2019_2024.csv');



function idx = localCumIndex(r, isLog)
    base = 100;
    if isLog
        % r are log returns: cum index = 100 * exp(cumsum(r))
        idx = base * exp(cumsum(r));
    else
        % r are simple returns: cum index = 100 * cumprod(1+r)
        idx = base * cumprod(1 + r);
    end
end
