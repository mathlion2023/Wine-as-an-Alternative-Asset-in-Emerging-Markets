%% SharpeComparison_RiskParity_vs_EqualWeight_18.m (peer-review fix, R3/S3)
% Devil's Advocate / Methodology reviewer finding: Table 8's ranking claim
% ("Risk Parity attains the highest Sharpe ratio") is not tested for
% significance against Equal-Weight, which is statistically
% indistinguishable in the reported point estimates (0.591 vs 0.591,
% rebalanced, LONG panel). This script applies the same 500-rep
% moving-block bootstrap (block=6 months) already used in Appendix C.6
% to the Sharpe-ratio DIFFERENCE between these two constructions'
% realized rebalanced return series.
clear; clc;

load('MasterData_USD_2013_2024.mat', 'M');
names = M.Properties.VariableNames(2:end);
N = numel(names);
dates = M.Period(2:end);
T = numel(dates);

load('PIT_WineReturns_18.mat', 'pitWineRet', 'cutoffSet', 'cutoffPos', 'wineIdx');

window = 36; rebalEvery = 3; costRate = 0.005; kappaAnn = 0.02;
lb = zeros(N,1); ub = ones(N,1);
fmopts = optimoptions('fmincon','Display','off','MaxFunctionEvaluations',2e5,'MaxIterations',5e3,'Algorithm','sqp');
portTypes = ["RiskParity","EqualWeight"];
nP = numel(portTypes);

decisionIdx = window:rebalEvery:T;
nDec = numel(decisionIdx);
evalIdx = decisionIdx(1):T;

wealthRebal   = ones(T,nP);
turnoverHist  = zeros(nDec,nP);
wCurrent      = ones(N,nP)/N;
wealthRebal(decisionIdx(1)-1,:) = 1;

for d = 1:nDec
    tau = decisionIdx(d);
    Rw_tau = pitWineRet{cutoffPos(tau)};
    win = (tau-window+1):tau;
    Rw  = Rw_tau(win,:);
    Sigma = ledoitWolfShrink(Rw);

    for k = 1:nP
        prevW = wCurrent(:,k);
        switch portTypes(k)
            case "RiskParity",  w = solveRiskParity(Sigma, N, lb, ub, fmopts);
            case "EqualWeight", w = ones(N,1)/N;
        end
        turnoverHist(d,k) = sum(abs(w - prevW));
        wCurrent(:,k) = w;
    end

    nextTau = min(tau + rebalEvery, T);
    for k = 1:nP
        wv = wealthRebal(tau,k);
        wv = wv * (1 - costRate*turnoverHist(d,k));
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

wealth = wealthRebal(evalIdx,:);
ret = diff(wealth) ./ wealth(1:end-1,:);     % simple returns, both series
logret = log(1+ret);

sharpeRP  = mean(logret(:,1))/std(logret(:,1)) * sqrt(12)/sqrt(12); % annualization cancels in the ratio below; keep monthly units for the bootstrap, report annualized at the end
sharpeEW  = mean(logret(:,2))/std(logret(:,2));
fprintf('Point estimates (monthly-units Sharpe): RiskParity=%.4f, EqualWeight=%.4f, diff=%.4f\n', ...
    sharpeRP, sharpeEW, sharpeRP-sharpeEW);
fprintf('Annualized Sharpe (sqrt(12) scaling applies equally to both, cancels in diff t-test): RiskParity=%.3f, EqualWeight=%.3f\n', ...
    sharpeRP*sqrt(12), sharpeEW*sqrt(12));

%% Moving-block bootstrap on the Sharpe-ratio difference (same design as Appendix C.6)
nBoot = 500; blockLen = 6;
nObs = size(logret,1);
nBlocks = ceil(nObs/blockLen);
diffBoot = nan(nBoot,1);
rng(42); % fixed seed for reproducibility of this check
for b = 1:nBoot
    idx = [];
    while numel(idx) < nObs
        start = randi(nObs - blockLen + 1);
        idx = [idx, start:(start+blockLen-1)]; %#ok<AGROW>
    end
    idx = idx(1:nObs);
    rb = logret(idx,:);
    sRP = mean(rb(:,1))/std(rb(:,1));
    sEW = mean(rb(:,2))/std(rb(:,2));
    diffBoot(b) = sRP - sEW;
end
ci90 = prctile(diffBoot, [5 95]);
fprintf('Bootstrap 90%% CI on (RiskParity - EqualWeight) Sharpe difference (monthly units): [%.4f, %.4f]\n', ci90(1), ci90(2));
fprintf('CI crosses zero: %d (1=yes, not significant; 0=no, significant)\n', (ci90(1)<0 && ci90(2)>0));

result = table(sharpeRP*sqrt(12), sharpeEW*sqrt(12), (sharpeRP-sharpeEW)*sqrt(12), ci90(1)*sqrt(12), ci90(2)*sqrt(12), ...
    'VariableNames', {'Sharpe_RiskParity','Sharpe_EqualWeight','DeltaSharpe_Annualized','CI90_Lower','CI90_Upper'});
disp(result);
writetable(result, 'tab_sharpe_comparison_RP_vs_EW_18.csv');
fprintf('Saved tab_sharpe_comparison_RP_vs_EW_18.csv\n');

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
