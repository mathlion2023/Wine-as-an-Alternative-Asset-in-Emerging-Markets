%% Rebalance_vs_BuyHold_Bootstrap.m (2026-11-06, ARS panel R5)
% -------------------------------------------------------------------------
% Methodology reviewer finding (ARS panel): Table 7 / Online Appendix
% Tables S13-S14 report rebalanced-vs-buy-and-hold Sharpe/CumRet
% comparisons with no significance test attached, unlike the adjacent
% with-vs-without-wine comparison (Table 8), even though the headline
% claim ("rebalancing underperforms buy-and-hold for most constructions")
% is explicitly attributed in the figure captions to a single large event
% (the December 2023 Argentine peso devaluation).
%
% This script reuses the SAME realized backtest return series already
% computed by Compare_WithWithoutWine_18.m / _4country.m (the "WITH wine"
% half of each -- identical window/rebalEvery/costRate/solvers to
% Portfolio_Backtest_PointInTime_*.m, so these ARE Table 7's and Tables
% S13-S14's own realized monthly returns, not a new backtest) and applies
% the same moving-block bootstrap already used for the with/without-wine
% Sharpe difference (Table 8) to the PAIRED rebalanced-vs-buy-and-hold
% Sharpe/CumRet difference per construction, on both panels.
% -------------------------------------------------------------------------
clear; clc;

panels = struct( ...
    'tag',        {'18','4country'}, ...
    'masterFile', {'MasterData_USD_2013_2024.mat', 'MasterData_USD_2019_2024_4country.mat'}, ...
    'pitFile',    {'PIT_WineReturns_18.mat', 'PIT_WineReturns_4country.mat'}, ...
    'wineKeys',   {{'AUS_wine_USD','ARG_wine_USD','CHL_wine_USD'}, ...
                   {'SA_wine_USD','AUS_wine_USD','ARG_wine_USD','CHL_wine_USD'}});

window = 36; rebalEvery = 3; costRate = 0.005; kappaAnn = 0.02;
qpopts = optimoptions('quadprog','Display','off');
fmopts = optimoptions('fmincon','Display','off','MaxFunctionEvaluations',2e5,'MaxIterations',5e3,'Algorithm','sqp');
lpopts = optimoptions('linprog','Display','off');
portTypes = ["GMV","Tangency","RiskParity","CVaR","EqualWeight"];
nP = numel(portTypes);
rng(11); Bboot = 1000; blockLen = 6;

for p = 1:numel(panels)
    pan = panels(p);
    fprintf('\n=== Panel %s ===\n', pan.tag);
    load(pan.masterFile, 'M');
    names = M.Properties.VariableNames(2:end);
    N_all = numel(names);
    lvl = M{:, names};
    R_full = diff(log(lvl));
    [T, ~] = size(R_full);
    wineIdx = find(ismember(names, pan.wineKeys));

    S_pit = load(pan.pitFile, 'pitWineRet', 'cutoffPos');
    [retRebal, retBH] = runBacktest(R_full, N_all, zeros(N_all,1), ones(N_all,1), ...
        window, rebalEvery, costRate, kappaAnn, portTypes, qpopts, fmopts, lpopts, ...
        S_pit.pitWineRet, S_pit.cutoffPos, wineIdx);

    results = table();
    for k = 1:nP
        rR = retRebal(:,k); rB = retBH(:,k);
        [shR, cumR] = perfStats(rR); [shB, cumB] = perfStats(rB);
        dSharpe = shR - shB; dCum = cumR - cumB;

        nM = numel(rR);
        dShBoot = nan(Bboot,1); dCumBoot = nan(Bboot,1);
        nBlocks = ceil(nM/blockLen);
        for b = 1:Bboot
            starts = randi(nM-blockLen+1, nBlocks, 1);
            idx = []; for j = 1:nBlocks, idx = [idx, starts(j):(starts(j)+blockLen-1)]; end %#ok<AGROW>
            idx = idx(1:nM);
            [shRb,cumRb] = perfStats(rR(idx)); [shBb,cumBb] = perfStats(rB(idx));
            dShBoot(b) = shRb - shBb; dCumBoot(b) = cumRb - cumBb;
        end
        loSh = prctile(dShBoot,5); hiSh = prctile(dShBoot,95);
        loCum = prctile(dCumBoot,5); hiCum = prctile(dCumBoot,95);

        fprintf('%-12s Sharpe(Rebal)=%.3f Sharpe(BH)=%.3f dSharpe=%.3f [%.3f,%.3f] | CumRet(Rebal)=%.3f CumRet(BH)=%.3f dCum=%.3f [%.3f,%.3f]\n', ...
            portTypes(k), shR, shB, dSharpe, loSh, hiSh, cumR, cumB, dCum, loCum, hiCum);

        results = [results; table(portTypes(k), shR, shB, dSharpe, loSh, hiSh, cumR, cumB, dCum, loCum, hiCum, ...
            'VariableNames', {'Portfolio','Sharpe_Rebal','Sharpe_BH','dSharpe','dSharpe_p05','dSharpe_p95', ...
                              'CumRet_Rebal','CumRet_BH','dCumRet','dCumRet_p05','dCumRet_p95'})]; %#ok<AGROW>
    end
    outname = sprintf('tab_rebal_vs_buyhold_bootstrap_%s.csv', pan.tag);
    writetable(results, outname);
    fprintf('Saved %s\n', outname);
end

%% ================= Local functions (identical to Compare_WithWithoutWine_*.m's runBacktest) =================
function [sh, cumRet] = perfStats(ret)
    wealth = cumprod(1+ret);
    logret = log(1 + ret);
    annMean = mean(logret)*12; annVol = std(logret)*sqrt(12);
    sh = annMean/annVol;
    cumRet = wealth(end)-1;
end

function [retRebal, retBH] = runBacktest(R, N, lb, ub, window, rebalEvery, costRate, ...
    kappaAnn, portTypes, qpopts, fmopts, lpopts, pitWineRet, cutoffPos, wineIdxForPIT)
    [T,~] = size(R); nP = numel(portTypes);
    decisionIdx = window:rebalEvery:T; nDec = numel(decisionIdx);
    wealthRebal = ones(T,nP); turnoverHist = zeros(nDec,nP);
    wCurrent = ones(N,nP)/N; wealthRebal(decisionIdx(1)-1,:) = 1; w0BuyHold = [];
    for d = 1:nDec
        tau = decisionIdx(d);
        Rtau = pitWineRet{cutoffPos(tau)};
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
