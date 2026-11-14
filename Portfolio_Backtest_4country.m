%% Portfolio_Backtest_18.m
% -------------------------------------------------------------------------
% Rolling 36-month, quarterly-rebalanced portfolio backtest across all 18
% series in the USD master dataset, matching QF_Paper.pdf Appendix A.4's
% stated methodology for Tables 11-12 (2026-08-25, per user instruction).
%
% At each quarterly decision date tau: estimate (mu_tau, Sigma_tau) from the
% trailing 36 months (Ledoit-Wolf shrinkage toward a constant-correlation
% target), solve each of the six programs, apply the new weights, pay
% proportional transaction costs on traded notional and a wine carry cost,
% then let weights drift with realized returns until the next decision
% date. Buy&Hold sets weights once (at the first decision date) and never
% rebalances or pays costs again.
%
% ASSUMPTIONS NOT SPECIFIED IN THE REPO/MANUSCRIPT TEXT (flagged, not
% silently chosen for the reader to miss):
%   - Uniform transaction cost c_i = 0.5% of traded notional (matches the
%     default already used in portfolio_paths_tc.m).
%   - Wine storage/insurance carry cost kappa = 2%/year (12/month), applied
%     only to the 4 wine series (SA/AUS/ARG/CHL), consistent with the
%     Masset & Weisskopf (2015) real-fund cost range CLAUDE.md already
%     cites re: Referee 1's fixed-supply/perishability point. NOT a value
%     taken from this manuscript -- confirm/replace before reporting.
%   - Turnover-regularized mean-variance: lambda_mu = lambda_TO = 1 (still
%     unspecified in the paper), but this version DOES use the genuine
%     previous-period weight (unlike the static single-period proxy run
%     earlier), since a rolling backtest actually has one.
%   - BL_Tangency: equal-weight market, no active views (same proxy as
%     before -- still unspecified in the paper) -- this collapses to
%     equal-weight at every rebalance by construction (see chat), so its
%     backtest is mechanically identical to a passive equal-weight
%     portfolio. Included for completeness, not as a real BL result.
%   - No wine-specific liquidity caps (w_i <= w_bar_wine) are applied --
%     the paper mentions these but gives no numeric value; long-only
%     0<=w<=1 is the only bound enforced.
% -------------------------------------------------------------------------

clear; clc;
load('MasterData_USD_2019_2024_4country.mat', 'M');
names = M.Properties.VariableNames(2:end);
N = numel(names);
lvl = M{:, names};
R = diff(log(lvl));            % T x N monthly log-returns
dates = M.Period(2:end);       % T x 1, aligned to R
[T, ~] = size(R);

wineIdx = find(ismember(names, {'SA_wine_USD','AUS_wine_USD','ARG_wine_USD','CHL_wine_USD'}));

%% ---- Settings ----
window     = 36;               % rolling estimation window (months)
rebalEvery = 3;                 % quarterly
costRate   = 0.005;             % 0.5% proportional transaction cost
kappaAnn   = 0.02;              % 2%/year wine storage/insurance carry cost (ASSUMPTION)
lambda_mu  = 1; lambda_TO = 1;  % turnover-penalty coefficients (ASSUMPTION)

lb = zeros(N,1); ub = ones(N,1);
qpopts = optimoptions('quadprog','Display','off');
fmopts = optimoptions('fmincon','Display','off','MaxFunctionEvaluations',2e5,'MaxIterations',5e3,'Algorithm','sqp');
lpopts = optimoptions('linprog','Display','off');

portTypes = ["GMV","Tangency","RiskParity","CVaR","Turnoverpen","BL_Tangency"];
nP = numel(portTypes);

% BL_Tangency's implied risk-aversion delta is fixed ONCE from the
% full-sample equal-weight market (not re-estimated from each noisy
% 36-month trailing window): a rolling trailing mean is negative often
% enough that a per-window delta flips sign, which breaks the
% reverse-optimization identity (equilibrium recovers w_mkt only for
% delta>0) and makes the tangency-equivalent QP infeasible. This mirrors
% how delta is used in practice (a long-run risk-premium calibration, not
% a quarter-to-quarter re-estimate).
w_mkt = ones(N,1)/N;
Sigma_full = ledoitWolfShrink(R);
delta_fixed = (w_mkt'*mean(R)') / (w_mkt'*Sigma_full*w_mkt);
fprintf('BL_Tangency fixed delta (full-sample implied risk aversion): %.4f\n', delta_fixed);

decisionIdx = window:rebalEvery:T;   % month indices (in R) where we re-optimize
nDec = numel(decisionIdx);

%% ---- Rolling backtest: rebalanced version ----
wealthRebal   = ones(T,nP);
turnoverHist  = zeros(nDec,nP);
wPrev         = ones(N,nP)/N;         % start from equal-weight before any decision
wCurrent      = ones(N,nP)/N;
wealthRebal(decisionIdx(1)-1,:) = 1;  % normalize start just before first decision

w0BuyHold = [];                        % weights fixed at first decision, for Buy&Hold

for d = 1:nDec
    tau = decisionIdx(d);
    win = (tau-window+1):tau;
    Rw  = R(win,:);
    mu  = mean(Rw)';
    Sigma = ledoitWolfShrink(Rw);

    for k = 1:nP
        prevW = wCurrent(:,k);
        switch portTypes(k)
            case "GMV"
                w = solveGMV(Sigma, N, lb, ub, qpopts);
            case "Tangency"
                w = solveTangency(mu, Sigma, N, qpopts);
            case "RiskParity"
                w = solveRiskParity(Sigma, N, lb, ub, fmopts);
            case "CVaR"
                w = solveCVaR(Rw, N, lb, ub, lpopts);
            case "Turnoverpen"
                w = solveTurnoverpen(mu, Sigma, prevW, lambda_mu, lambda_TO, N, lb, ub, qpopts);
            case "BL_Tangency"
                w = solveBLTangency(Sigma, delta_fixed, N, qpopts);   % equal-weight market, no views
        end
        turnoverHist(d,k) = sum(abs(w - prevW));
        if d == 1
            w0BuyHold(:,k) = w; %#ok<AGROW>
        end
        wCurrent(:,k) = w;
    end

    % Apply this quarter's target weights from tau+1 to the next decision
    % date (or end of sample), paying transaction + wine carry costs.
    nextTau = min(tau + rebalEvery, T);
    for k = 1:nP
        wv = wealthRebal(tau,k);
        wv = wv * (1 - costRate*turnoverHist(d,k));   % pay cost on traded notional
        w  = wCurrent(:,k);
        for t = (tau+1):nextTau
            % BUG FIX (2026-10-09, peer-review-flagged): R is log returns;
            % "1 + R*w" approximates exp(r)-1 and materially understates
            % large moves. Wealth must compound on simple-return relatives.
            rel = exp(R(t,:))';                 % true gross simple-return relative per asset
            grossPort = w' * rel;                % 1 + portfolio simple return
            wineCarry = (kappaAnn/12) * sum(w(wineIdx));
            wv = wv * grossPort * (1 - wineCarry);
            wealthRebal(t,k) = wv;
            % drift weights with realized relatives
            w = (w .* rel) / (w'*rel);
        end
        wCurrent(:,k) = w;   % carry drifted weight into next quarter's prevW
    end
end

%% ---- Buy&Hold: set once at first decision date, never rebalance (no
% transaction costs after the initial allocation) -- but storage/insurance
% is a physical holding cost that accrues every month regardless of
% trading, so it MUST still apply here.
% BUG FIX (2026-10-09, peer-review-flagged): this loop previously never
% referenced kappaAnn at all, so Buy&Hold paid zero wine carry cost for
% the entire holding period. Also fixes the log-vs-simple-return bug.
startIdx = decisionIdx(1);
wealthBH = ones(T,nP);
wealthBH(startIdx,:) = 1;
for k = 1:nP
    w = w0BuyHold(:,k);
    for t = (startIdx+1):T
        rel = exp(R(t,:))';
        grossPort = w' * rel;
        wineCarry = (kappaAnn/12) * sum(w(wineIdx));
        wealthBH(t,k) = wealthBH(t-1,k) * grossPort * (1 - wineCarry);
        w = (w .* rel) / (w'*rel);
    end
end

%% ---- Performance metrics over the evaluated window ----
evalIdx = startIdx:T;
summRebal = summarize(wealthRebal(evalIdx,:), R(evalIdx,:), portTypes, mean(turnoverHist)');
summBH    = summarize(wealthBH(evalIdx,:),    R(evalIdx,:), portTypes, zeros(nP,1));

disp('==== Table 11: Rebalanced (quarterly, 36-mo rolling window, with costs) ====');
disp(summRebal);
writetable(summRebal, 'tab_portfolio_rebal_4country.csv');

disp('==== Table 12: Buy & Hold (weights set at first decision date) ====');
disp(summBH);
writetable(summBH, 'tab_portfolio_buyhold_4country.csv');

fprintf('Backtest window: %s to %s (%d evaluated months, %d rebalances)\n', ...
    datestr(dates(startIdx),'yyyy-mm'), datestr(dates(end),'yyyy-mm'), numel(evalIdx), nDec);

%% ---- Figures 4-6 (2026-09-03): index paths, rebalanced base=100 at start
% of the evaluated (out-of-sample) window. No prior script plotted/saved
% these for the 18-series backtest -- the old FixedWeight.eps/BHWeight.eps
% in this folder are from the original 9-asset, 2019-2024 pipeline.
idxRebal = 100 * wealthRebal(evalIdx,:) ./ wealthRebal(evalIdx(1),:);
idxBH    = 100 * wealthBH(evalIdx,:)    ./ wealthBH(evalIdx(1),:);
evalDates = dates(evalIdx);

f4 = figure('Name','Rebalanced (fixed weights)','Position',[100 100 800 500]);
plot(evalDates, idxRebal, 'LineWidth', 1.3);
grid on; datetick('x','yyyy','keeplimits'); xlabel('Date'); ylabel('Index (base = 100)');
legend(cellstr(portTypes), 'Location', 'northwest');
title('Rebalanced (quarterly, 36-mo rolling window, with costs)');
exportgraphics(f4, 'FixedWeight_4country.png', 'Resolution', 200);
fprintf('Saved FixedWeight_4country.png\n');

f5 = figure('Name','Buy&Hold (drifting weights)','Position',[100 100 800 500]);
plot(evalDates, idxBH, 'LineWidth', 1.3);
grid on; datetick('x','yyyy','keeplimits'); xlabel('Date'); ylabel('Index (base = 100)');
legend(cellstr(portTypes), 'Location', 'northwest');
title('Buy&Hold (weights fixed at first decision date)');
exportgraphics(f5, 'BHWeight_4country.png', 'Resolution', 200);
fprintf('Saved BHWeight_4country.png\n');

%% ================= Helper functions =================

function Sigma = ledoitWolfShrink(R)
    [Tn,N] = size(R);
    S = cov(R);
    sd = sqrt(diag(S));
    Rcorr = S ./ (sd*sd');
    rbar = (sum(Rcorr(:)) - N) / (N*(N-1));
    F = rbar * (sd*sd');
    F(1:N+1:end) = diag(S);
    Xc = R - mean(R);
    piMat = zeros(N);
    for i = 1:N
        for j = 1:N
            piMat(i,j) = mean((Xc(:,i).*Xc(:,j) - S(i,j)).^2);
        end
    end
    piHat = sum(piMat(:));
    gammaHat = sum((F(:) - S(:)).^2);
    shrinkage = max(0, min(1, (piHat/gammaHat) / Tn));
    Sigma = shrinkage*F + (1-shrinkage)*S;
end

function w = solveGMV(Sigma, N, lb, ub, qpopts)
    w = quadprog(2*Sigma, zeros(N,1), [], [], ones(1,N), 1, lb, ub, [], qpopts);
end

function w = solveTangency(mu, Sigma, N, qpopts)
    y0 = ones(N,1)/N;
    y = quadprog(2*Sigma, zeros(N,1), [], [], mu', 1, zeros(N,1), [], y0, qpopts);
    w = y / sum(y);
end

function w = solveRiskParity(Sigma, N, lb, ub, fmopts)
    b = ones(N,1)/N;
    obj = @(w) logERCObjective(w, Sigma, b);
    w0 = ones(N,1)/N;
    w = fmincon(obj, w0, [], [], ones(1,N), 1, lb, ub, [], fmopts);
end

function val = logERCObjective(w, Sigma, b)
    sp = sqrt(w'*Sigma*w);
    RC = max(w .* (Sigma*w) / sp, 1e-12);
    N = numel(w); val = 0;
    for i = 1:N-1
        for j = i+1:N
            val = val + (log(RC(i)) - log(b(i)) - log(RC(j)) + log(b(j)))^2;
        end
    end
end

function w = solveCVaR(R, N, lb, ub, lpopts)
    alpha = 0.95;
    Tn = size(R,1);
    f = [zeros(N,1); 1; ones(Tn,1)/((1-alpha)*Tn)];
    A = [-R, -ones(Tn,1), -eye(Tn)];
    b = zeros(Tn,1);
    Aeq = [ones(1,N), 0, zeros(1,Tn)]; beq = 1;
    lbv = [lb; -Inf; zeros(Tn,1)]; ubv = [ub; Inf; Inf(Tn,1)];
    x = linprog(f, A, b, Aeq, beq, lbv, ubv, lpopts);
    w = x(1:N);
end

function w = solveTurnoverpen(mu, Sigma, prevW, lambda_mu, lambda_TO, N, lb, ub, qpopts)
    H = blkdiag(2*Sigma, zeros(N));
    f = [-lambda_mu*mu; lambda_TO*ones(N,1)];
    A = [eye(N), -eye(N); -eye(N), -eye(N)];
    b = [prevW; -prevW];
    Aeq = [ones(1,N), zeros(1,N)]; beq = 1;
    lbv = [lb; zeros(N,1)]; ubv = [ub; Inf(N,1)];
    x = quadprog(H, f, A, b, Aeq, beq, lbv, ubv, [], qpopts);
    w = x(1:N);
end

function w = solveBLTangency(Sigma, delta, N, qpopts)
    w_mkt = ones(N,1)/N;
    pi_bl = delta * Sigma * w_mkt;
    y0 = ones(N,1)/N;
    y = quadprog(2*Sigma, zeros(N,1), [], [], pi_bl', 1, zeros(N,1), [], y0, qpopts);
    w = y / sum(y);
end

function T = summarize(wealth, R, portTypes, avgTurnover)
    nP = size(wealth,2);
    ret = diff(wealth) ./ wealth(1:end-1,:);
    annMean = mean(ret)'*12;
    annVol  = std(ret)'*sqrt(12);
    sharpe  = annMean ./ annVol;
    cvar95  = nan(nP,1);
    for k = 1:nP
        r = sort(ret(:,k));
        m = max(1, round(0.05*numel(r)));
        cvar95(k) = -mean(r(1:m));
    end
    cumRet = wealth(end,:)' - 1;
    maxDD = nan(nP,1);
    for k = 1:nP
        peak = cummax(wealth(:,k));
        dd = wealth(:,k)./peak - 1;
        maxDD(k) = min(dd);
    end
    T = table(portTypes', cumRet, annMean, annVol, sharpe, cvar95, maxDD, avgTurnover, ...
        'VariableNames', {'Portfolio','CumRet','AnnMean','AnnVol','Sharpe','CVaR95','MaxDrawdown','AvgTurnover'});
end
