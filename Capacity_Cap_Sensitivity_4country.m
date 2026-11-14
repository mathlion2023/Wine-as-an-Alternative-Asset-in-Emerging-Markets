%% Capacity_Cap_Sensitivity_4country.m (2026-10-11, peer-review-panel item)
% -------------------------------------------------------------------------
% Section 4.4 states that no wine-specific capacity cap is calibrated or
% enforced in Table 10/11/12's reported results -- the long-only [0,1] box
% constraint is uniform across every asset including wine. A reviewer asked
% for at least one illustrative capacity-constrained re-solve, rather than
% leaving the capacity discussion purely qualitative.
%
% This script re-solves GMV, Tangency, Risk Parity, and CVaR (identical
% data, shrinkage covariance, and solvers as Portfolio_Optimization_4country.m)
% with one additional constraint: each of the four wine series (SAFW, AUSFW,
% ARGFW, CHIFW) is capped at a uniform, deliberately illustrative w_i <= 2%,
% rather than a value individually calibrated to each country's own realized
% market turnover (Section 4.4 reports turnover differs by two orders of
% magnitude across the four countries, Australia ~$102,400/month vs.
% Argentina ~$100/month -- a single uniform cap is not claimed to be
% country-specific capacity-accurate, only a demonstration of mechanics and
% direction of effect). Output: tab_capacity_cap_sensitivity_4country.csv,
% comparing uncapped (Table 10) vs. 2%-capped weights side by side.
% -------------------------------------------------------------------------

clear; clc;
load('MasterData_USD_2019_2024_4country.mat', 'M');
names = M.Properties.VariableNames(2:end);
N = numel(names);
lvl = M{:, names};
R = diff(log(lvl));
[Tn, ~] = size(R);
mu = mean(R)';

wineKeys = {'SAFW','AUSFW','ARGFW','CHIFW'};
wineColNames = {'SA_wine_USD','AUS_wine_USD','ARG_wine_USD','CHL_wine_USD'};
wineIdx = zeros(1,4);
for w_ = 1:4
    idx_ = find(strcmp(names, wineColNames{w_}));
    assert(isscalar(idx_), 'Wine column %s not found.', wineColNames{w_});
    wineIdx(w_) = idx_;
end
clear w_ idx_

%% ---- Ledoit-Wolf shrinkage toward a constant-correlation target (identical to Portfolio_Optimization_4country.m) ----
S = cov(R);
sd = sqrt(diag(S));
Rcorr = S ./ (sd*sd');
rbar = (sum(Rcorr(:)) - N) / (N*(N-1));
F = rbar * (sd*sd');
F(1:N+1:end) = diag(S);
Xc = R - mean(R);
T2 = Tn;
piMat = zeros(N);
for i = 1:N
    for j = 1:N
        piMat(i,j) = mean((Xc(:,i).*Xc(:,j) - S(i,j)).^2);
    end
end
piHat = sum(piMat(:));
gammaHat = sum((F(:) - S(:)).^2);
kappaHat = piHat / gammaHat;
shrinkage = max(0, min(1, kappaHat / T2));
Sigma = shrinkage*F + (1-shrinkage)*S;

Aeq = ones(1,N); beq = 1;
qpopts = optimoptions('quadprog','Display','off');
fmopts = optimoptions('fmincon','Display','off','MaxFunctionEvaluations',2e5,'MaxIterations',5e3,'Algorithm','sqp');
lpopts = optimoptions('linprog','Display','off');

capLevels = [1.00, 0.02];  % uncapped (replicates Table 10) vs. 2% uniform wine cap
results = struct();

for ci = 1:numel(capLevels)
    cap = capLevels(ci);
    lb = zeros(N,1); ub = ones(N,1);
    ub(wineIdx) = min(ub(wineIdx), cap);

    % GMV
    w_gmv = quadprog(2*Sigma, zeros(N,1), [], [], Aeq, beq, lb, ub, [], qpopts);

    % Tangency: when uncapped, use the exact QP y-substitution (identical to
    % Portfolio_Optimization_4country.m, so this row reproduces Table 10
    % exactly) -- the substitution does not support a general box constraint
    % on w once a cap binds, so the capped case is solved by direct SQP
    % maximization of the Sharpe ratio in w-space instead.
    if cap >= 1
        y0 = ones(N,1)/N;
        y = quadprog(2*Sigma, zeros(N,1), [], [], mu', 1, zeros(N,1), [], y0, qpopts);
        w_tan = y / sum(y);
    else
        sharpeObj = @(w) -(w'*mu) / sqrt(w'*Sigma*w);
        w_tan = fmincon(sharpeObj, ones(N,1)/N, [], [], Aeq, beq, lb, ub, [], fmopts);
    end

    % Risk Parity
    b = ones(N,1)/N;
    ircObj = @(w) logERCObjective(w, Sigma, b);
    w_rp = fmincon(ircObj, ones(N,1)/N, [], [], Aeq, beq, lb, ub, [], fmopts);

    % CVaR
    alpha = 0.95;
    f = [zeros(N,1); 1; ones(Tn,1)/((1-alpha)*Tn)];
    Acvar = [-R, -ones(Tn,1), -eye(Tn)];
    bcvar = zeros(Tn,1);
    Aeq2 = [ones(1,N), 0, zeros(1,Tn)];
    beq2 = 1;
    lb2 = [lb; -Inf; zeros(Tn,1)];
    ub2 = [ub;  Inf; Inf(Tn,1)];
    xcvar = linprog(f, Acvar, bcvar, Aeq2, beq2, lb2, ub2, lpopts);
    w_cvar = xcvar(1:N);

    results(ci).cap = cap;
    results(ci).W = [w_gmv, w_tan, w_rp, w_cvar];
end

Wunc = results(1).W; Wcap = results(2).W;
wineNamesOut = wineKeys';
Tout = table(string(names)', Wunc(:,1), Wcap(:,1), Wunc(:,2), Wcap(:,2), Wunc(:,3), Wcap(:,3), Wunc(:,4), Wcap(:,4), ...
    'VariableNames', {'Series','GMV_uncapped','GMV_2pctcap','Tangency_uncapped','Tangency_2pctcap', ...
                       'RiskParity_uncapped','RiskParity_2pctcap','CVaR_uncapped','CVaR_2pctcap'});
disp(Tout);
writetable(Tout, 'tab_capacity_cap_sensitivity_4country.csv');

fprintf('\nWine total allocation (sum of 4 wine series), uncapped vs. 2pct-capped:\n');
constructionNames = {'GMV','Tangency','RiskParity','CVaR'};
for k = 1:4
    fprintf('  %-12s uncapped=%.1f%%  capped=%.1f%%\n', constructionNames{k}, ...
        100*sum(Wunc(wineIdx,k)), 100*sum(Wcap(wineIdx,k)));
end

function val = logERCObjective(w, Sigma, b)
    sp = sqrt(w'*Sigma*w);
    RC = w .* (Sigma*w) / sp;
    RC = max(RC, 1e-12);
    N = numel(w);
    val = 0;
    for i = 1:N-1
        for j = i+1:N
            val = val + (log(RC(i)) - log(b(i)) - log(RC(j)) + log(b(j)))^2;
        end
    end
end
