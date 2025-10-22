%% portfolio_paths_tc.m
% Cumulative performance (2019–2024) under:
%   (A) periodic rebalancing with proportional transaction costs
%   (B) buy-and-hold (no rebalancing, no costs)
%
% Requirements in workspace before running:
%   - R: T x 9 monthly returns (simple or log)
%   - dates: T x 1 datetime
%
% Column order of R:
% ["SAfw","Ausfw","Argfw","chifw","SP500","MSCIEM","Bond","Gold","Liv_ex100"]

clear; clc;
load('projectdata.mat');
load('projectdate.mat'); % T×1 datetime

% Data=table2array(data);
R   = diff(log(data)); % T×N returns

%% ===== USER SETTINGS =====
isLogReturn   = true;     % true if R are log-returns; false if simple returns
toAnnual      = 12;        % monthly data


% Proportional transaction cost (round-trip-like % on traded notional)
% e.g., 0.5% = 0.005 ; use smaller for large caps, larger for niche assets.
costRate      = 0.005;

% Rebal frequency for the “rebalancing” simulation: 'monthly','quarterly','annual'
rebalanceFreq = 'monthly';

% Fixed weights (from your table, as decimals summing to 1)
W = struct();
W.GMV          = [ 1  ,  4,  2,  3, 11, 11, 22, 19, 26 ]'/100;
W.Tangency     = [ 1  ,  4,  2,  2, 20, 15, 10, 27, 21 ]'/100;
W.RiskParity   = [ 1  ,  1,  1,  1, 44,  1, 30, 19,  1 ]'/100;
W.CVaR         = [ 1  ,  1,  1,  0,  9,  0,  9, 26, 54 ]'/100;
W.Turnoverpen  = [ 1  ,  4,  2,  2, 19, 15, 11, 26, 20 ]'/100;
W.BL_Tangency  = [ 0  ,  2,  2,  0, 32, 19,  0, 37,  7 ]'/100;

portNames = fieldnames(W);

%% ===== SANITY CHECKS =====
dates  = date(2:end);
[T, N] = size(R);
if N ~= 9
    error('R must be T x 9.');
end
if numel(dates) ~= T
    error('dates must have T rows to match R.');
end



good   = all(isfinite(R),2);
R      = R(good,:);
dates  = dates(good);
T      = size(R,1);

% Ensure each weight vector sums to 1 and is long-only
% for k = 1:numel(portNames)
%     s = portNames{k};
%     if abs(sum(W.(s)) - 1) > 1e-10
%         error('Weights for %s do not sum to 1.', s);
%     end
%     if any(W.(s) < -1e-12)
%         error('Negative weight in %s (expected long-only).', s);
%     end
% end

% Convert to simple returns if input is in logs (we simulate wealth in simple space)
if isLogReturn
    G = exp(R);                 % gross relatives
    Rsim = G - 1;               % simple returns
else
    Rsim = R;                   % already simple returns
end
Gsim = 1 + Rsim;                % gross relatives (T x N)

%% ===== RUN TWO MODES FOR EACH PORTFOLIO =====
OUT = struct();   % will hold paths + stats

for k = 1:numel(portNames)
    name = portNames{k};
    w0   = W.(name);

    % (A) Periodic rebalancing with transaction costs
    OUT.(name).rebal = simulateWithRebalCosts(Gsim, dates, w0, rebalanceFreq, costRate);

    % (B) Buy-and-hold (no rebalancing, no costs)
    OUT.(name).buyhold = simulateBuyAndHold(Gsim, dates, w0);
end

%% ===== SUMMARIES =====
summ = table('Size',[numel(portNames)*2 6], ...
    'VariableTypes',["string","string","double","double","double","double"], ...
    'VariableNames',["Portfolio","Mode","CumRet","AnnMean","AnnVol","Sharpe"]);

row = 0;
for k = 1:numel(portNames)
    name = portNames{k};

    % A) Rebalance + costs
    [CR, AM, AV, SR] = pathStats(OUT.(name).rebal.wealth, Rsim, OUT.(name).rebal.turnover, toAnnual);
    row = row + 1;
    summ(row,:) = {string(name), "Rebalance+TC", CR, AM, AV, SR};

    % B) Buy-and-hold
    [CR, AM, AV, SR] = pathStats(OUT.(name).buyhold.wealth, Rsim, zeros(T,1), toAnnual);
    row = row + 1;
    summ(row,:) = {string(name), "Buy&Hold", CR, AM, AV, SR};
end

disp('==== Cumulative and annualized stats (2019–2024) ====');
disp(summ);

%% ===== PLOTS =====
% 1) Wealth indices
% figure('Color','w'); tiledlayout(2,1,'TileSpacing','compact');
% nexttile; hold on; box on;
% for k = 1:numel(portNames)
%     plot(dates, 100*OUT.(portNames{k}).rebal.wealth, 'LineWidth', 1.4, ...
%         'DisplayName', [portNames{k} ' (reb+TC)']);
% end
% yline(100,'k:'); grid on;
% title(sprintf('Wealth Index (base=100): Rebalance (%s) with TC=%.2f%%', rebalanceFreq, 100*costRate));
% xlabel('Date'); ylabel('Index');
% legend('Location','bestoutside');

%nexttile; hold on; box on;
hold on
for k = 1:numel(portNames)
    plot(dates, 100*OUT.(portNames{k}).buyhold.wealth, 'LineWidth', 1.4, ...
        'DisplayName', [portNames{k} ' (buy&hold)']);
end
%yline(100,'k:'); grid on;
%title('Wealth Index (base=100): Buy-and-Hold (no costs)');
xlabel('Date'); ylabel('Index');
legend('Location','best');grid on
datetick
axis tight

% 2) Bar chart of cumulative return by mode
figure('Color','w'); hold on; box on;
cats = categorical(string(summ.Portfolio) + " - " + string(summ.Mode));
bar(cats, summ.CumRet);
ylabel('Cumulative Return (decimal)');
title('Total Cumulative Return: Rebalance+TC vs Buy-and-Hold');
grid on;


%% ===================== LOCAL FUNCTIONS =====================

function out = simulateWithRebalCosts(G, dates, wTarget, freq, costRate)
% Simulate wealth path with periodic rebalancing to wTarget and proportional costs
% G: T x N gross relatives (1 + simple return)
% dates: T x 1 datetime
% wTarget: N x 1 target weights (sum=1)
% freq: 'monthly'|'quarterly'|'annual'
% costRate: proportional cost per dollar traded (e.g., 0.005 = 50 bps)

[T,N] = size(G);
wealth = zeros(T,1);
wealth(1) = 1.0;

% Rebalance schedule (indices in 1..T)
rebIdx = false(T,1);
switch lower(freq)
    case 'monthly'
        rebIdx(:) = true;
    case 'quarterly'
        [yy,mm] = ymd(dates);
        rebIdx = ismember(mm, [3,6,9,12]);
    case 'annual'
        [yy,mm] = ymd(dates);
        rebIdx = (mm == 12);
    otherwise
        error('Unknown rebalance frequency.');
end

% Start fully invested at target; pay initial cost for entering positions
w = wTarget;  % post-trade weights at t=1 (after initial trade)
initCost = costRate * sum(abs(w));   % entering from 0 to w
W0 = 1 - initCost;
wealth(1) = W0 * (w' * G(1,:)');     % apply first month growth after initial trade cost
turnover = zeros(T,1);
turnover(1) = sum(abs(w));           % report initial turnover (from cash to invested)

for t = 2:T
    % 1) Drift weights with returns (pre-trade weights)
    w_pre = (w .* G(t-1,:).');
    w_pre = w_pre / sum(w_pre);      % normalize to sum to 1

    % 2) Wealth evolves by pre-trade portfolio return
    grossPort = sum(w .* G(t-1,:).');  % previous post-trade weights times last period gross
    wealth(t) = wealth(t-1) * grossPort;

    % 3) If rebalance date, trade to target and pay costs today (before next period)
    if rebIdx(t)
        delta = wTarget - w_pre;                      % desired - current
        tradedNotional = wealth(t) * sum(abs(delta)); % dollars traded
        tc = costRate * tradedNotional;
        wealth(t) = max(wealth(t) - tc, eps);         % deduct costs
        w = wTarget;                                  % set new post-trade weights
        turnover(t) = sum(abs(delta));
    else
        w = w_pre;                                    % no trade (carry pre-trade forward)
        turnover(t) = 0;
    end
end

% Apply last month gross return
wealth(T) = wealth(T) * sum(w .* G(T,:).');

out.wealth   = wealth;          % path of wealth (starts at ~1 - initCost, base ~1)
out.turnover = turnover;        % L1 turnover each decision date
out.dates    = dates;
end


function out = simulateBuyAndHold(G, dates, w0)
% Simulate buy-and-hold (no rebalancing, no transaction costs)
% Start with weights w0 (sum=1), let them drift

[T,N] = size(G);
wealth = zeros(T,1); wealth(1) = 1.0;

% Period 1 wealth
wealth(1) = wealth(1) * (w0' * G(1,:)');

% Track evolving weights (optional)
w = w0;

for t = 2:T
    grossPort = sum(w .* G(t,:).');
    wealth(t) = wealth(t-1) * grossPort;

    % Update weights by asset relatives (no trading)
    w = (w .* G(t,:).');
    w = w / sum(w);
end

out.wealth = wealth; out.dates = dates; out.w = w;
end


function [cumRet, annMean, annVol, sharpe] = pathStats(wealth, Rsim, turnover, toAnnual)
% Wealth is a path; convert to returns and compute stats.
% r_f = 0

ret = [NaN; wealth(2:end)./wealth(1:end-1) - 1];
ret = ret(~isnan(ret));

m  = mean(ret,'omitnan');
sd = std(ret,0,'omitnan');

cumRet  = wealth(end) - 1;
annMean = (1 + m)^toAnnual - 1;
annVol  = sd * sqrt(toAnnual);

if annVol > 0
    sharpe = annMean / annVol;
else
    sharpe = NaN;
end
end
