%% WithWithoutWine_HighCost_18.m (2026-10-28, co-author Faff feedback)
% -------------------------------------------------------------------------
% Robert Faff's point: Table S19's LONG-panel CVaR row reports a baseline
% with-minus-without-wine Sharpe difference of 0.237 (90% CI [0.01, 0.38]).
% TransactionCost_Sensitivity_18.m reports that under a wine-specific 20%
% round-trip cost (vs. the 0.5% baseline), CVaR's WITH-wine Sharpe falls
% from 0.39 to 0.14. Subtracting naively (0.39 - 0.237 = 0.153 as an
% implied WITHOUT-wine Sharpe, compared against 0.14 WITH-wine under the
% higher cost) suggests the positive incremental benefit of wine reverses
% under the higher cost. That subtraction is not obviously wrong -- the
% WITHOUT-wine portfolio holds no wine, so its Sharpe is mechanically
% invariant to the wine-cost assumption, and comparing it against the
% WITH-wine portfolio under the higher cost is a legitimate like-for-like
% comparison of the SAME two portfolios under the SAME realized dates and
% non-wine-leg costs. But a point-estimate subtraction does not by itself
% establish whether the resulting difference is distinguishable from noise.
% This script runs the actual comparison directly (not by subtracting two
% already-rounded point estimates) and bootstraps it, using the same
% moving-block procedure as Compare_WithWithoutWine_18.m.
% -------------------------------------------------------------------------
clear; clc;
load('MasterData_USD_2013_2024.mat', 'M');
names = M.Properties.VariableNames(2:end);
N_all = numel(names);
lvl = M{:, names};
R_full = diff(log(lvl));
dates = M.Period(2:end);
[T, ~] = size(R_full);
wineIdx = find(ismember(names, {'AUS_wine_USD','ARG_wine_USD','CHL_wine_USD'}));
noWineIdx = setdiff(1:N_all, wineIdx);

load('PIT_WineReturns_18.mat', 'pitWineRet', 'cutoffPos');

window = 36; rebalEvery = 3; kappaAnn = 0.02;
costRateBase = 0.005;
costRateWineHigh = 0.20;
portTypes = ["GMV","Tangency","RiskParity","CVaR","EqualWeight"];
nP = numel(portTypes);
qpopts = optimoptions('quadprog','Display','off');
fmopts = optimoptions('fmincon','Display','off','MaxFunctionEvaluations',2e5,'MaxIterations',5e3,'Algorithm','sqp');
lpopts = optimoptions('linprog','Display','off');

decisionIdx = window:rebalEvery:T; nDec = numel(decisionIdx);
wineMask = false(1,N_all); wineMask(wineIdx) = true;

%% ---- WITH wine, high wine-leg cost (20% on wine turnover, 0.5% elsewhere) ----
wealthRebal = ones(T,nP); turnoverHist = zeros(nDec,nP); wCurrent = ones(N_all,nP)/N_all;
wealthRebal(decisionIdx(1)-1,:) = 1;
for d = 1:nDec
    tau = decisionIdx(d);
    Rtau = pitWineRet{cutoffPos(tau)};
    win = (tau-window+1):tau; Rw = Rtau(win,:);
    mu = mean(Rw)'; Sigma = ledoitWolfShrink(Rw);
    costThisRebal = zeros(nP,1);
    for k = 1:nP
        prevW = wCurrent(:,k);
        switch portTypes(k)
            case "GMV",         w = solveGMV(Sigma, N_all, zeros(N_all,1), ones(N_all,1), qpopts);
            case "Tangency",    w = solveTangency(mu, Sigma, N_all, qpopts);
            case "RiskParity",  w = solveRiskParity(Sigma, N_all, zeros(N_all,1), ones(N_all,1), fmopts);
            case "CVaR",        w = solveCVaR(Rw, N_all, zeros(N_all,1), ones(N_all,1), lpopts);
            case "EqualWeight", w = ones(N_all,1)/N_all;
        end
        dw = abs(w - prevW);
        turnoverWine = sum(dw(wineMask)); turnoverNonWine = sum(dw(~wineMask));
        turnoverHist(d,k) = turnoverWine + turnoverNonWine;
        costThisRebal(k) = costRateBase*turnoverNonWine + costRateWineHigh*turnoverWine;
        wCurrent(:,k) = w;
        if d == 1, w0BuyHold(:,k) = w; end %#ok<AGROW>
    end
    nextTau = min(tau + rebalEvery, T);
    for k = 1:nP
        wv = wealthRebal(tau,k); wv = wv * (1 - costThisRebal(k)); w = wCurrent(:,k);
        for t = (tau+1):nextTau
            Rt = pitWineRet{cutoffPos(t)}; rel = exp(Rt(t,:))'; grossPort = w'*rel;
            wineCarry = (kappaAnn/12) * sum(w(wineIdx));
            wv = wv * grossPort * (1 - wineCarry); wealthRebal(t,k) = wv;
            w = (w .* rel) / (w'*rel);
        end
        wCurrent(:,k) = w;
    end
end
evalIdx = decisionIdx(1):T;
retRebalWithHigh = diff(wealthRebal(evalIdx,:)) ./ wealthRebal(evalIdx(1:end-1),:);

%% ---- WITHOUT wine, baseline cost (mechanically invariant to the wine-cost
% assumption since no wine is held -- reuse runBacktest from
% Compare_WithWithoutWine_18.m's logic directly here for a self-contained
% script) ----
R_nowine = R_full(:, noWineIdx); N_nw = numel(noWineIdx);
[retRebalNo, ~] = runBacktestNoWine(R_nowine, N_nw, window, rebalEvery, costRateBase, portTypes, qpopts, fmopts, lpopts);

%% ---- Compare under the SAME dates, block-bootstrap the Sharpe difference ----
rng(7); Bboot = 1000; blockLen = 6;
results = table();
for k = 1:nP
    rW = retRebalWithHigh(:,k); rN = retRebalNo(:,k);
    [shW, cumW] = perfStats(rW); [shN, cumN] = perfStats(rN);
    dSharpe = shW - shN; dCum = cumW - cumN;
    nM = numel(rW); nBlocks = ceil(nM/blockLen);
    dShBoot = nan(Bboot,1);
    for b = 1:Bboot
        starts = randi(nM-blockLen+1, nBlocks, 1);
        idx = []; for j = 1:nBlocks, idx = [idx, starts(j):(starts(j)+blockLen-1)]; end %#ok<AGROW>
        idx = idx(1:nM);
        [shWb,~] = perfStats(rW(idx)); [shNb,~] = perfStats(rN(idx));
        dShBoot(b) = shWb - shNb;
    end
    loSh = prctile(dShBoot,5); hiSh = prctile(dShBoot,95);
    results = [results; table(portTypes(k), shW, shN, dSharpe, loSh, hiSh, cumW, cumN, dCum, ...
        'VariableNames', {'Portfolio','Sharpe_With_HighCost','Sharpe_Without','dSharpe','dSharpe_p05','dSharpe_p95','CumRet_With_HighCost','CumRet_Without','dCumRet'})]; %#ok<AGROW>
end
disp(results);
writetable(results, 'tab_withwithoutwine_highcost_18.csv');
fprintf('Saved tab_withwithoutwine_highcost_18.csv\n');

%% ================= Local functions =================
function [sh, cumRet] = perfStats(ret)
    wealth = cumprod(1+ret);
    logret = log(1 + ret);
    annMean = mean(logret)*12; annVol = std(logret)*sqrt(12);
    sh = annMean/annVol; cumRet = wealth(end)-1;
end

function [retRebal, retBH] = runBacktestNoWine(R, N, window, rebalEvery, costRate, portTypes, qpopts, fmopts, lpopts)
    [T,~] = size(R); nP = numel(portTypes);
    decisionIdx = window:rebalEvery:T; nDec = numel(decisionIdx);
    wealthRebal = ones(T,nP); turnoverHist = zeros(nDec,nP);
    wCurrent = ones(N,nP)/N; wealthRebal(decisionIdx(1)-1,:) = 1; w0BuyHold = [];
    for d = 1:nDec
        tau = decisionIdx(d);
        win = (tau-window+1):tau; Rw = R(win,:);
        mu = mean(Rw)'; Sigma = ledoitWolfShrink(Rw);
        for k = 1:nP
            prevW = wCurrent(:,k);
            switch portTypes(k)
                case "GMV",         w = solveGMV(Sigma, N, zeros(N,1), ones(N,1), qpopts);
                case "Tangency",    w = solveTangency(mu, Sigma, N, qpopts);
                case "RiskParity",  w = solveRiskParity(Sigma, N, zeros(N,1), ones(N,1), fmopts);
                case "CVaR",        w = solveCVaR(Rw, N, zeros(N,1), ones(N,1), lpopts);
                case "EqualWeight", w = ones(N,1)/N;
            end
            turnoverHist(d,k) = sum(abs(w-prevW));
            if d==1, w0BuyHold(:,k) = w; end %#ok<AGROW>
            wCurrent(:,k) = w;
        end
        nextTau = min(tau+rebalEvery, T);
        for k = 1:nP
            wv = wealthRebal(tau,k); wv = wv*(1-costRate*turnoverHist(d,k)); w = wCurrent(:,k);
            for t = (tau+1):nextTau
                rel = exp(R(t,:))'; grossPort = w'*rel;
                wv = wv*grossPort; wealthRebal(t,k) = wv;
                w = (w.*rel)/(w'*rel);
            end
            wCurrent(:,k) = w;
        end
    end
    startIdx = decisionIdx(1); wealthBH = ones(T,nP); wealthBH(startIdx,:) = 1;
    for k = 1:nP
        w = w0BuyHold(:,k);
        for t = (startIdx+1):T
            rel = exp(R(t,:))'; grossPort = w'*rel;
            wealthBH(t,k) = wealthBH(t-1,k)*grossPort;
            w = (w.*rel)/(w'*rel);
        end
    end
    evalIdx2 = startIdx:T;
    retRebal = diff(wealthRebal(evalIdx2,:)) ./ wealthRebal(evalIdx2(1:end-1),:);
    retBH    = diff(wealthBH(evalIdx2,:))    ./ wealthBH(evalIdx2(1:end-1),:);
end

function Sigma = ledoitWolfShrink(R)
    [Tn,N] = size(R); S = cov(R); sd = sqrt(diag(S)); Rcorr = S./(sd*sd');
    rbar = (sum(Rcorr(:))-N)/(N*(N-1)); F = rbar*(sd*sd'); F(1:N+1:end) = diag(S);
    Xc = R-mean(R); piMat = zeros(N);
    for i=1:N, for j=1:N, piMat(i,j) = mean((Xc(:,i).*Xc(:,j)-S(i,j)).^2); end, end
    piHat = sum(piMat(:)); gammaHat = sum((F(:)-S(:)).^2);
    shrinkage = max(0,min(1,(piHat/gammaHat)/Tn)); Sigma = shrinkage*F+(1-shrinkage)*S;
end
function w = solveGMV(Sigma, N, lb, ub, qpopts)
    w = quadprog(2*Sigma, zeros(N,1), [], [], ones(1,N), 1, lb, ub, [], qpopts);
end
function w = solveTangency(mu, Sigma, N, qpopts)
    y0 = ones(N,1)/N; y = quadprog(2*Sigma, zeros(N,1), [], [], mu', 1, zeros(N,1), [], y0, qpopts);
    w = y/sum(y);
end
function w = solveRiskParity(Sigma, N, lb, ub, fmopts)
    b = ones(N,1)/N; obj = @(w) logERCObjective(w, Sigma, b); w0 = ones(N,1)/N;
    w = fmincon(obj, w0, [], [], ones(1,N), 1, lb, ub, [], fmopts);
end
function val = logERCObjective(w, Sigma, b)
    sp = sqrt(w'*Sigma*w); RC = max(w.*(Sigma*w)/sp, 1e-12); N = numel(w); val = 0;
    for i=1:N-1, for j=i+1:N, val = val + (log(RC(i))-log(b(i))-log(RC(j))+log(b(j)))^2; end, end
end
function w = solveCVaR(R, N, lb, ub, lpopts)
    alpha = 0.95; Tn = size(R,1);
    f = [zeros(N,1); 1; ones(Tn,1)/((1-alpha)*Tn)];
    A = [-R, -ones(Tn,1), -eye(Tn)]; b = zeros(Tn,1);
    Aeq = [ones(1,N), 0, zeros(1,Tn)]; beq = 1;
    lbv = [lb; -Inf; zeros(Tn,1)]; ubv = [ub; Inf; Inf(Tn,1)];
    x = linprog(f, A, b, Aeq, beq, lbv, ubv, lpopts); w = x(1:N);
end
