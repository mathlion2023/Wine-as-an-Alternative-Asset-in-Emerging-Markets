%% Compare_WithWithoutWine_4country.m (2026-10-10, peer-review item 8)
% -------------------------------------------------------------------------
% Peer review item 8: "Positive wine weights under risk parity are partly
% mechanical when every included asset receives a positive risk budget.
% They do not establish incremental investment value. Compare otherwise
% identical portfolios with and without regional wine using common dates,
% constraints, and costs, and report uncertainty around net performance
% and downside-risk differences."
%
% This script runs the SAME rolling 36-month, quarterly-rebalanced
% backtest (same dates, same solvers, same transaction cost) twice on the
% SHORT/baseline panel: once WITH the four wine series available to the
% optimizer (reusing the point-in-time, no-look-ahead wine returns already
% built and cached by Portfolio_Backtest_PointInTime_4country.m -- Task 7
% -- rather than rebuilding them), and once with wine EXCLUDED from the
% investable universe entirely (N=14 benchmarks only, no wine carry cost
% since none is held). Reports, per construction: Sharpe/CumRet/MaxDD with
% vs. without wine, the difference, and a moving-block bootstrap 90% CI on
% the Sharpe and CumRet differences, using the REALIZED monthly portfolio
% return series from the single actual backtest run (weights are NOT
% re-optimized under each bootstrap draw -- a standard, tractable
% simplification, the same one implicitly used by Jobson-Korkie-style
% Sharpe-difference tests on a fixed pair of return series).
% -------------------------------------------------------------------------

clear; clc;
load('MasterData_USD_2019_2024_4country.mat', 'M');
names = M.Properties.VariableNames(2:end);
N_all = numel(names);
lvl = M{:, names};
R_full = diff(log(lvl));
dates = M.Period(2:end);
[T, ~] = size(R_full);
wineIdx = find(ismember(names, {'SA_wine_USD','AUS_wine_USD','ARG_wine_USD','CHL_wine_USD'}));
noWineIdx = setdiff(1:N_all, wineIdx);

% Turnoverpen and BL_Tangency REMOVED 2026-10-10 (peer-review item 9, per
% author decision) -- see Portfolio_Backtest_PointInTime_4country.m's
% header note. EqualWeight added as the naive comparator, matching the
% headline Tables 10-12.
window = 36; rebalEvery = 3; costRate = 0.005; kappaAnn = 0.02;
qpopts = optimoptions('quadprog','Display','off');
fmopts = optimoptions('fmincon','Display','off','MaxFunctionEvaluations',2e5,'MaxIterations',5e3,'Algorithm','sqp');
lpopts = optimoptions('linprog','Display','off');
portTypes = ["GMV","Tangency","RiskParity","CVaR","EqualWeight"];
nP = numel(portTypes);
decisionIdx = window:rebalEvery:T; nDec = numel(decisionIdx);
evalIdx = decisionIdx(1):T;

%% ---- WITH wine: reuse Task 7's cached point-in-time wine returns ----
S_pit = load('PIT_WineReturns_4country.mat', 'pitWineRet', 'cutoffSet', 'cutoffPos');
[retRebalWith, retBHWith] = runBacktest(R_full, wineIdx, N_all, lb_(N_all), ub_(N_all), ...
    window, rebalEvery, costRate, kappaAnn, portTypes, qpopts, fmopts, lpopts, ...
    S_pit.pitWineRet, S_pit.cutoffPos, wineIdx);

%% ---- WITHOUT wine: 14-series universe, no point-in-time complexity needed
% (benchmarks have no estimation look-ahead), no wine carry cost ----
R_nowine = R_full(:, noWineIdx);
N_nw = numel(noWineIdx);
[retRebalNo, retBHNo] = runBacktest(R_nowine, [], N_nw, lb_(N_nw), ub_(N_nw), ...
    window, rebalEvery, costRate, kappaAnn, portTypes, qpopts, fmopts, lpopts, ...
    {}, [], []);

%% ---- Compare: Sharpe/CumRet/MaxDD with vs without, + block-bootstrap CI on the difference ----
rng(7); Bboot = 1000; blockLen = 6;
results = table();
for k = 1:nP
    % retRebalWith/retRebalNo are already restricted to the evaluation
    % window by runBacktest (both use the same T/decisionIdx/evalIdx,
    % since the two universes share the same date grid) -- no further
    % indexing needed here.
    rW = retRebalWith(:,k);
    rN = retRebalNo(:,k);
    [shW, cumW, ddW] = perfStats(rW);
    [shN, cumN, ddN] = perfStats(rN);
    dSharpe = shW - shN; dCum = cumW - cumN;

    nM = numel(rW);
    dShBoot = nan(Bboot,1); dCumBoot = nan(Bboot,1);
    nBlocks = ceil(nM/blockLen);
    for b = 1:Bboot
        starts = randi(nM-blockLen+1, nBlocks, 1);
        idx = []; for j = 1:nBlocks, idx = [idx, starts(j):(starts(j)+blockLen-1)]; end %#ok<AGROW>
        idx = idx(1:nM);
        [shWb,cumWb,~] = perfStats(rW(idx));
        [shNb,cumNb,~] = perfStats(rN(idx));
        dShBoot(b) = shWb - shNb; dCumBoot(b) = cumWb - cumNb;
    end
    loSh = prctile(dShBoot,5); hiSh = prctile(dShBoot,95);
    loCum = prctile(dCumBoot,5); hiCum = prctile(dCumBoot,95);

    results = [results; table(portTypes(k), shW, shN, dSharpe, loSh, hiSh, cumW, cumN, dCum, loCum, hiCum, ...
        'VariableNames', {'Portfolio','Sharpe_With','Sharpe_Without','dSharpe','dSharpe_p05','dSharpe_p95', ...
                          'CumRet_With','CumRet_Without','dCumRet','dCumRet_p05','dCumRet_p95'})]; %#ok<AGROW>
end
disp(results);
writetable(results, 'tab_withwithoutwine_4country.csv');
fprintf('Saved tab_withwithoutwine_4country.csv\n');

%% ================= Local functions =================
function v = lb_(n), v = zeros(n,1); end
function v = ub_(n), v = ones(n,1); end

function [sh, cumRet, maxDD] = perfStats(ret)
    wealth = cumprod(1+ret);
    % BUG FIX (2026-10-19, co-author Faff flagged pre-resubmission): see the
    % identical fix in Portfolio_Backtest_PointInTime_18.m -- Appendix A
    % states Sharpe/volatility use log(1+R_{p,t+1}), not the simple ret
    % this previously used directly. CumRet/MaxDrawdown still use simple-
    % return-compounded wealth, consistent with the text.
    logret = log(1 + ret);
    annMean = mean(logret)*12; annVol = std(logret)*sqrt(12);
    sh = annMean/annVol;
    cumRet = wealth(end)-1;
    peak = cummax(wealth); dd = wealth./peak - 1; maxDD = min(dd);
end

function [retRebal, retBH] = runBacktest(R, wineIdxLocal, N, lb, ub, window, rebalEvery, costRate, ...
    kappaAnn, portTypes, qpopts, fmopts, lpopts, pitWineRet, cutoffPos, wineIdxForPIT)
    [T,~] = size(R); nP = numel(portTypes);
    usePIT = ~isempty(pitWineRet);
    decisionIdx = window:rebalEvery:T; nDec = numel(decisionIdx);
    wealthRebal = ones(T,nP); turnoverHist = zeros(nDec,nP);
    wCurrent = ones(N,nP)/N; wealthRebal(decisionIdx(1)-1,:) = 1; w0BuyHold = [];
    for d = 1:nDec
        tau = decisionIdx(d);
        if usePIT, Rtau = pitWineRet{cutoffPos(tau)}; else, Rtau = R; end
        win = (tau-window+1):tau; Rw = Rtau(win,:);
        mu = mean(Rw)'; Sigma = ledoitWolfShrink(Rw);
        for k = 1:nP
            prevW = wCurrent(:,k);
            switch portTypes(k)
                case "GMV",         w = solveGMV(Sigma, N, lb, ub, qpopts);
                case "Tangency",    w = solveTangency(mu, Sigma, N, qpopts);
                case "RiskParity",  w = solveRiskParity(Sigma, N, lb, ub, fmopts);
                case "CVaR",        w = solveCVaR(Rw, N, lb, ub, lpopts);
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
                if usePIT, Rt = pitWineRet{cutoffPos(t)}; else, Rt = R; end
                rel = exp(Rt(t,:))'; grossPort = w'*rel;
                wineCarry = 0; if ~isempty(wineIdxForPIT), wineCarry = (kappaAnn/12)*sum(w(wineIdxForPIT)); end
                wv = wv*grossPort*(1-wineCarry); wealthRebal(t,k) = wv;
                w = (w.*rel)/(w'*rel);
            end
            wCurrent(:,k) = w;
        end
    end
    startIdx = decisionIdx(1); wealthBH = ones(T,nP); wealthBH(startIdx,:) = 1;
    for k = 1:nP
        w = w0BuyHold(:,k);
        for t = (startIdx+1):T
            if usePIT, Rt = pitWineRet{cutoffPos(t)}; else, Rt = R; end
            rel = exp(Rt(t,:))'; grossPort = w'*rel;
            wineCarry = 0; if ~isempty(wineIdxForPIT), wineCarry = (kappaAnn/12)*sum(w(wineIdxForPIT)); end
            wealthBH(t,k) = wealthBH(t-1,k)*grossPort*(1-wineCarry);
            w = (w.*rel)/(w'*rel);
        end
    end
    evalIdx = startIdx:T;
    retRebal = diff(wealthRebal(evalIdx,:)) ./ wealthRebal(evalIdx(1:end-1),:);
    retBH    = diff(wealthBH(evalIdx,:))    ./ wealthBH(evalIdx(1:end-1),:);
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
