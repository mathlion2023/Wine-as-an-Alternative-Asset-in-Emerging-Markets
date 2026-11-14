%% CapacityCap_RollingBacktest_4country.m (2026-11-06, ARS panel R4)
% -------------------------------------------------------------------------
% Perspective / Devil's Advocate finding: the illustrative 2% wine-cap
% sensitivity check (Appendix A.3 / Capacity_Cap_Sensitivity_4country.m)
% is applied only to Table 6's static, in-sample allocation, never to the
% rolling point-in-time backtest that generates Table 7 / S13-S14's
% headline realized Sharpe numbers -- exactly where a binding capacity
% constraint would matter most. This script re-runs the SHORT/baseline
% panel's rolling 36-month backtest for Risk Parity and CVaR (the two
% largest wine-weight constructions) with the same uniform 2% per-wine-
% series cap applied at every rebalance date, and compares realized
% Sharpe/CumRet against the uncapped backtest already reported.
% -------------------------------------------------------------------------
clear; clc;
load('MasterData_USD_2019_2024_4country.mat', 'M');
names = M.Properties.VariableNames(2:end);
N_all = numel(names);
lvl = M{:, names};
R_full = diff(log(lvl));
[T, ~] = size(R_full);
wineIdx = find(ismember(names, {'SA_wine_USD','AUS_wine_USD','ARG_wine_USD','CHL_wine_USD'}));

window = 36; rebalEvery = 3; costRate = 0.005; kappaAnn = 0.02;
fmopts = optimoptions('fmincon','Display','off','MaxFunctionEvaluations',2e5,'MaxIterations',5e3,'Algorithm','sqp');
lpopts = optimoptions('linprog','Display','off');
portTypes = ["RiskParity","CVaR"];
nP = numel(portTypes);

S_pit = load('PIT_WineReturns_4country.mat', 'pitWineRet', 'cutoffPos');

capLevels = [1.00, 0.02];
allResults = table();
for ci = 1:numel(capLevels)
    cap = capLevels(ci);
    lb = zeros(N_all,1); ub = ones(N_all,1); ub(wineIdx) = min(ub(wineIdx), cap);
    [retRebal, retBH] = runBacktest(R_full, N_all, lb, ub, window, rebalEvery, costRate, kappaAnn, ...
        portTypes, fmopts, lpopts, S_pit.pitWineRet, S_pit.cutoffPos, wineIdx);
    for k = 1:nP
        [shR, cumR] = perfStats(retRebal(:,k));
        [shB, cumB] = perfStats(retBH(:,k));
        tagCap = "uncapped"; if cap < 1, tagCap = "2pct_wine_cap"; end
        fprintf('[%s / %s] Rebal: Sharpe=%.3f CumRet=%.3f | BuyHold: Sharpe=%.3f CumRet=%.3f\n', ...
            portTypes(k), tagCap, shR, cumR, shB, cumB);
        allResults = [allResults; table(portTypes(k), tagCap, shR, cumR, shB, cumB, ...
            'VariableNames', {'Portfolio','Cap','Sharpe_Rebal','CumRet_Rebal','Sharpe_BH','CumRet_BH'})]; %#ok<AGROW>
    end
end
disp(allResults);
writetable(allResults, 'tab_capacitycap_rollingbacktest_4country.csv');
fprintf('Saved tab_capacitycap_rollingbacktest_4country.csv\n');

function [sh, cumRet] = perfStats(ret)
    wealth = cumprod(1+ret); logret = log(1+ret);
    sh = (mean(logret)*12)/(std(logret)*sqrt(12)); cumRet = wealth(end)-1;
end

function [retRebal, retBH] = runBacktest(R, N, lb, ub, window, rebalEvery, costRate, ...
    kappaAnn, portTypes, fmopts, lpopts, pitWineRet, cutoffPos, wineIdxForPIT)
    [T,~] = size(R); nP = numel(portTypes);
    decisionIdx = window:rebalEvery:T; nDec = numel(decisionIdx);
    wealthRebal = ones(T,nP); turnoverHist = zeros(nDec,nP);
    wCurrent = ones(N,nP)/N; wealthRebal(decisionIdx(1)-1,:) = 1; w0BuyHold = [];
    for d = 1:nDec
        tau = decisionIdx(d);
        Rtau = pitWineRet{cutoffPos(tau)};
        win = (tau-window+1):tau; Rw = Rtau(win,:);
        Sigma = ledoitWolfShrink(Rw);
        for k = 1:nP
            prevW = wCurrent(:,k);
            switch portTypes(k)
                case "RiskParity", w = solveRiskParity(Sigma, N, lb, ub, fmopts);
                case "CVaR",       w = solveCVaR(Rw, N, lb, ub, lpopts);
            end
            turnoverHist(d,k) = sum(abs(w-prevW));
            if d==1, w0BuyHold(:,k) = w; end %#ok<AGROW>
            wCurrent(:,k) = w;
        end
        nextTau = min(tau+rebalEvery, T);
        for k = 1:nP
            wv = wealthRebal(tau,k); wv = wv*(1-costRate*turnoverHist(d,k)); w = wCurrent(:,k);
            for t = (tau+1):nextTau
                Rt = pitWineRet{cutoffPos(t)};
                rel = exp(Rt(t,:))'; grossPort = w'*rel;
                wineCarry = (kappaAnn/12)*sum(w(wineIdxForPIT));
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
            Rt = pitWineRet{cutoffPos(t)};
            rel = exp(Rt(t,:))'; grossPort = w'*rel;
            wineCarry = (kappaAnn/12)*sum(w(wineIdxForPIT));
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
