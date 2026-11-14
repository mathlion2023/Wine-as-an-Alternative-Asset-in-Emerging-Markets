%% Diagnose_MERVAL_Dec2023_4country.m (2026-10-19)
% SHORT/baseline (18-series, 4-country) panel counterpart of
% Diagnose_MERVAL_Dec2023.m -- see that script's header for rationale.
% Read-only: reuses cached PIT_WineReturns_4country.mat, re-runs only the
% backtest loop (not the expensive point-in-time reconstruction), and
% reports Tangency's MERVAL weight/contribution around Dec 2023.
clear; clc;

load('MasterData_USD_2019_2024_4country.mat','M');
names = M.Properties.VariableNames(2:end);
N = numel(names);
mervalCol = find(strcmp(names,'MERVAL_USD'));
assert(isscalar(mervalCol));

load('PIT_WineReturns_4country.mat', 'pitWineRet', 'cutoffSet', 'cutoffPos', 'dates', 'wineIdx');
window = 36; rebalEvery = 3; costRate = 0.005; kappaAnn = 0.02;
lb = zeros(N,1); ub = ones(N,1);
qpopts = optimoptions('quadprog','Display','off');
fmopts = optimoptions('fmincon','Display','off','MaxFunctionEvaluations',2e5,'MaxIterations',5e3,'Algorithm','sqp');
lpopts = optimoptions('linprog','Display','off');
portTypes = ["GMV","Tangency","RiskParity","CVaR","EqualWeight"];
nP = numel(portTypes);
T = size(pitWineRet{1},1);

decisionIdx = window:rebalEvery:T;
nDec = numel(decisionIdx);

wealthRebal = ones(T,nP);
turnoverHist = zeros(nDec,nP);
wCurrent = ones(N,nP)/N;
wealthRebal(decisionIdx(1)-1,:) = 1;
mervalWeightHist = nan(nDec,nP);

for d = 1:nDec
    tau = decisionIdx(d);
    Rw_tau = pitWineRet{cutoffPos(tau)};
    win = (tau-window+1):tau;
    Rw = Rw_tau(win,:);
    mu = mean(Rw)';
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
        mervalWeightHist(d,k) = w(mervalCol);
        turnoverHist(d,k) = sum(abs(w - prevW));
        wCurrent(:,k) = w;
    end

    nextTau = min(tau + rebalEvery, T);
    for k = 1:nP
        wv = wealthRebal(tau,k);
        wv = wv * (1 - costRate*turnoverHist(d,k));
        w = wCurrent(:,k);
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

decDates = dates(decisionIdx);
nearIdx = find(decDates >= datetime(2023,6,1) & decDates <= datetime(2024,3,1));
fprintf('==== SHORT panel: MERVAL weight at decision dates near the event ====\n');
disp(array2table(mervalWeightHist(nearIdx,:), 'VariableNames', cellstr(portTypes), 'RowNames', cellstr(datestr(decDates(nearIdx),'yyyy-mm'))))

wIdx = find(dates >= datetime(2023,8,1) & dates <= datetime(2024,3,1));
fprintf('\n==== SHORT panel: wealth index level around the event ====\n');
disp(array2table(wealthRebal(wIdx,:), 'VariableNames', cellstr(portTypes), 'RowNames', cellstr(datestr(dates(wIdx),'yyyy-mm'))))

fprintf('\n==== SHORT panel: monthly return around the event ====\n');
retRebal = wealthRebal(wIdx(2:end),:) ./ wealthRebal(wIdx(1:end-1),:) - 1;
disp(array2table(retRebal, 'VariableNames', cellstr(portTypes), 'RowNames', cellstr(datestr(dates(wIdx(2:end)),'yyyy-mm'))))

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
