%% TransactionCost_Sensitivity_18.m (peer-review fix, R7)
% Perspective reviewer finding: the backtest applies a uniform 0.5%
% round-trip transaction cost to every asset including wine, but publicly
% posted buyer's premiums at the auction houses in this paper's own data
% are far higher (Langton's 18%, Sotheby's/Christie's ~24-25%). This
% script re-runs the LONG-panel point-in-time backtest (Table 8) with a
% wine-specific high transaction cost (20%, representative of the
% verified buyer's-premium range) applied only to wine-leg turnover, cost
% on non-wine legs unchanged at 0.5%, and reports the resulting Sharpe/
% CumRet deltas against the baseline. Loads the already-cached point-in-
% time wine-return reconstructions (PIT_WineReturns_18.mat) rather than
% re-running the expensive reconstruction step.
clear; clc;

load('MasterData_USD_2013_2024.mat', 'M');
names = M.Properties.VariableNames(2:end);
N = numel(names);
dates = M.Period(2:end);
T = numel(dates);

load('PIT_WineReturns_18.mat', 'pitWineRet', 'cutoffSet', 'cutoffPos', 'wineIdx');

window = 36; rebalEvery = 3; kappaAnn = 0.02;
costRateBase = 0.005;
costRateWineHigh = 0.20;   % representative of verified 18-25% buyer's premiums

lb = zeros(N,1); ub = ones(N,1);
qpopts = optimoptions('quadprog','Display','off');
fmopts = optimoptions('fmincon','Display','off','MaxFunctionEvaluations',2e5,'MaxIterations',5e3,'Algorithm','sqp');
lpopts = optimoptions('linprog','Display','off');
portTypes = ["GMV","Tangency","RiskParity","CVaR","EqualWeight"];
nP = numel(portTypes);

decisionIdx = window:rebalEvery:T;
nDec = numel(decisionIdx);
evalIdx = decisionIdx(1):T;

wineMask = false(1,N); wineMask(wineIdx) = true;

scenarios = struct('name', {'Baseline (0.5% uniform)', 'High wine cost (20% wine, 0.5% other)'}, ...
    'wineRate', {costRateBase, costRateWineHigh});

summaries = cell(numel(scenarios),1);

for s = 1:numel(scenarios)
    wineRate = scenarios(s).wineRate;

    wealthRebal   = ones(T,nP);
    turnoverHist  = zeros(nDec,nP);
    wCurrent      = ones(N,nP)/N;
    wealthRebal(decisionIdx(1)-1,:) = 1;

    for d = 1:nDec
        tau = decisionIdx(d);
        Rw_tau = pitWineRet{cutoffPos(tau)};
        win = (tau-window+1):tau;
        Rw  = Rw_tau(win,:);
        mu  = mean(Rw)';
        Sigma = ledoitWolfShrink(Rw);

        for k = 1:nP
            prevW = wCurrent(:,k);
            switch portTypes(k)
                case "GMV",         w = solveGMV(Sigma, N, lb, ub, qpopts);
                case "Tangency",    w = solveTangency(mu, Sigma, N, qpopts);
                case "RiskParity",  w = solveRiskParity(Sigma, N, lb, ub, fmopts);
                case "CVaR",        w = solveCVaR(Rw, N, lb, ub, lpopts);
                case "EqualWeight", w = ones(N,1)/N;
            end
            dw = abs(w - prevW);
            turnoverWine    = sum(dw(wineMask));
            turnoverNonWine = sum(dw(~wineMask));
            turnoverHist(d,k) = turnoverWine + turnoverNonWine;  % reported total, for comparability with baseline table
            costThisRebal = costRateBase*turnoverNonWine + wineRate*turnoverWine;
            wCurrent(:,k) = w;
            if d == 1, w0BuyHold(:,k) = w; end %#ok<AGROW>
            costStore(d,k) = costThisRebal; %#ok<AGROW>
        end

        nextTau = min(tau + rebalEvery, T);
        for k = 1:nP
            wv = wealthRebal(tau,k);
            wv = wv * (1 - costStore(d,k));
            w  = wCurrent(:,k);
            for t = (tau+1):nextTau
                Rt_pit = pitWineRet{cutoffPos(t)};
                rel = exp(Rt_pit(t,:))';
                grossPort = w' * rel;
                wineCarry = (kappaAnn/12) * sum(w(wineIdx));
                wv = wv * grossPort * (1 - wineCarry);
                wealthRebal(t,k) = wv;
                w = (w .* rel) / (w'*rel);
            end
            wCurrent(:,k) = w;
        end
    end

    evalRet = nan(numel(evalIdx), N);
    for ii = 1:numel(evalIdx)
        t = evalIdx(ii);
        Rt_pit = pitWineRet{cutoffPos(t)};
        evalRet(ii,:) = Rt_pit(t,:);
    end
    summaries{s} = summarize(wealthRebal(evalIdx,:), evalRet, portTypes, mean(turnoverHist)');
    fprintf('==== %s ====\n', scenarios(s).name);
    disp(summaries{s});
end

base = summaries{1}; high = summaries{2};
delta = table(base.Portfolio, base.Sharpe, high.Sharpe, high.Sharpe - base.Sharpe, base.CumRet, high.CumRet, ...
    'VariableNames', {'Portfolio','Sharpe_Baseline','Sharpe_HighWineCost','DeltaSharpe','CumRet_Baseline','CumRet_HighWineCost'});
disp(delta);
writetable(delta, 'tab_txncost_sensitivity_18.csv');
fprintf('Saved tab_txncost_sensitivity_18.csv\n');

%% ================= Helper functions (identical to Portfolio_Backtest_PointInTime_18.m) =================

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

function T = summarize(wealth, R, portTypes, avgTurnover)
    nP = size(wealth,2);
    ret = diff(wealth) ./ wealth(1:end-1,:);
    logret = log(1 + ret);
    annMean = mean(logret)'*12;
    annVol  = std(logret)'*sqrt(12);
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
