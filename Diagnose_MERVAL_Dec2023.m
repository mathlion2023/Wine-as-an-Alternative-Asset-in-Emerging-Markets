%% Diagnose_MERVAL_Dec2023.m (2026-10-19)
% -------------------------------------------------------------------------
% Co-author (Faff) flagged, before resubmission: "the reported [MERVAL]
% gain appears difficult to reconcile with the local index movement and
% peso depreciation." Re-derives MERVAL_USD's December 2023 return from
% raw inputs, reuses the already-cached PIT_WineReturns_18.mat (point-in-
% time wine reconstruction) to re-run the LONG-panel backtest's loop only
% (no re-reconstruction), and reports Tangency's MERVAL weight and the
% portfolio-level contribution around the event. Read-only diagnostic:
% does not overwrite any production .mat/.csv/.png output.
% -------------------------------------------------------------------------
clear; clc;

%% ---- 1. Hand-trace from raw Bloomberg inputs ----
optsFX = detectImportOptions('Wine DATA.xlsx', 'Sheet', 'Sheet1');
optsFX.VariableNamesRange = 'A6'; optsFX.DataRange = 'A7';
Tfx = readtable('Wine DATA.xlsx', optsFX);
Tfx.Properties.VariableNames{1} = 'Date';
Tfx.Date = datetime(Tfx.Date);
idx = find(Tfx.Date >= datetime(2023,9,1) & Tfx.Date <= datetime(2024,2,29));
raw = Tfx(idx, {'Date','MERVAL_TR','USDARS'});
raw.MERVAL_USD = raw.MERVAL_TR ./ raw.USDARS;
fprintf('==== Raw MERVAL trace (local index, USDARS fx quote, derived USD level) ====\n');
disp(raw)
r = diff(log(raw.MERVAL_USD));
fprintf('==== Log returns on MERVAL_USD (hand-derived) ====\n');
disp(table(raw.Date(2:end), r, 'VariableNames', {'Date','LogRet'}))

%% ---- 2. Confirm against the actual pipeline master dataset ----
load('MasterData_USD_2013_2024.mat','M');
idx2 = find(M.Period >= datetime(2023,9,1) & M.Period <= datetime(2024,2,1));
T2 = M(idx2, {'Period','MERVAL_USD'});
fprintf('\n==== MasterData_USD_2013_2024.mat MERVAL_USD (pipeline) ====\n');
disp(T2)
r2 = diff(log(T2.MERVAL_USD));
fprintf('==== Pipeline log returns ====\n');
disp(table(T2.Period(2:end), r2, 'VariableNames', {'Period','LogRet'}))

%% ---- 3. Re-run the LONG-panel rolling backtest loop (reuse cached PIT reconstruction) ----
load('PIT_WineReturns_18.mat', 'pitWineRet', 'cutoffSet', 'cutoffPos', 'dates', 'wineIdx');
names = M.Properties.VariableNames(2:end);
N = numel(names);
mervalCol = find(strcmp(names,'MERVAL_USD'));
assert(isscalar(mervalCol));

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

%% ---- 4. Report MERVAL weight and wealth path around Nov 2023 - Feb 2024 ----
decDates = dates(decisionIdx);
nearIdx = find(decDates >= datetime(2023,6,1) & decDates <= datetime(2024,3,1));
fprintf('\n==== Tangency/GMV/RiskParity/CVaR MERVAL weight at decision dates near the event ====\n');
disp(array2table(mervalWeightHist(nearIdx,:), 'VariableNames', cellstr(portTypes), 'RowNames', cellstr(datestr(decDates(nearIdx),'yyyy-mm'))))

wIdx = find(dates >= datetime(2023,8,1) & dates <= datetime(2024,3,1));
fprintf('\n==== Wealth index level (rebalanced, all 5 constructions) around the event ====\n');
disp(array2table(wealthRebal(wIdx,:), 'VariableNames', cellstr(portTypes), 'RowNames', cellstr(datestr(dates(wIdx),'yyyy-mm'))))

fprintf('\n==== Portfolio-level monthly return (rebalanced wealth) around the event ====\n');
retRebal = wealthRebal(wIdx(2:end),:) ./ wealthRebal(wIdx(1:end-1),:) - 1;
disp(array2table(retRebal, 'VariableNames', cellstr(portTypes), 'RowNames', cellstr(datestr(dates(wIdx(2:end)),'yyyy-mm'))))

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
