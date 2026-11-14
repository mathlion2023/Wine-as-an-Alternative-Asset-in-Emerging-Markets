%% Portfolio_Optimization_18.m
% -------------------------------------------------------------------------
% Static long-only portfolio weights across all 18 series in the USD
% master dataset (Build_MasterData_USD.m), matching Table 10's four
% fully-specified constructions from QF_Paper.pdf Appendix A.4 (2026-08-25,
% per user instruction to run portfolio optimization "for all 18"):
%   (i)   GMV        -- eq. (8):  min w'Sigma w  s.t. 1'w=1, 0<=w<=1
%   (ii)  Tangency   -- eq. (9):  max w'mu / sqrt(w'Sigma w)  s.t. 1'w=1, 0<=w<=1
%   (iii) RiskParity -- eq. (10): equal risk contribution (log-ERC), b_i=1/N
%   (iv)  CVaR       -- eq. (11): Rockafellar-Uryasev, alpha=0.95, historical scenarios
%
% NOT implemented here (need author-specified inputs not in the repo/paper
% text -- see chat): (v) Turnover-regularized mean-variance needs lambda_mu,
% lambda_TO and a previous-weight reference; (vi) BL_Tangency needs actual
% investor views (P,Q,Omega), market weights w^mkt, delta, tau.
%
% Uses the full Jan 2013-Nov 2024 sample (142 monthly log-returns), with a
% Ledoit-Wolf-type shrinkage covariance toward a constant-correlation
% target, per the paper's stated risk-model choice (Appendix A.4: "shrinkage
% covariance (Ledoit-Wolf to a diagonal or constant-correlation target)").
% -------------------------------------------------------------------------

clear; clc;
load('MasterData_USD_2013_2024.mat', 'M');
names = M.Properties.VariableNames(2:end);
N = numel(names);
lvl = M{:, names};
R = diff(log(lvl));           % T x N monthly log-returns
[Tn, ~] = size(R);
mu = mean(R)';                % N x 1

%% ---- Ledoit-Wolf shrinkage toward a constant-correlation target ----
S = cov(R);                                  % sample covariance
sd = sqrt(diag(S));
Rcorr = S ./ (sd*sd');
rbar = (sum(Rcorr(:)) - N) / (N*(N-1));       % average off-diagonal correlation
F = rbar * (sd*sd');
F(1:N+1:end) = diag(S);                       % constant-correlation target, same variances

% Ledoit-Wolf (2004) shrinkage intensity estimate
Xc = R - mean(R);
T2 = Tn;
piMat = zeros(N); rhoMat = zeros(N); gammaSum = 0;
for i = 1:N
    for j = 1:N
        piMat(i,j) = mean((Xc(:,i).*Xc(:,j) - S(i,j)).^2);
    end
end
piHat = sum(piMat(:));
gammaHat = sum((F(:) - S(:)).^2);
% rho (off-diagonal covariance-of-covariances term) set to 0 -- the
% simplified/robust variant commonly used when a closed-form rho is not
% derived for the constant-correlation target; still a valid shrinkage
% estimator, just not the exact asymptotic-optimal rho of Ledoit-Wolf (2003).
kappaHat = (piHat) / gammaHat;
shrinkage = max(0, min(1, kappaHat / T2));
Sigma = shrinkage*F + (1-shrinkage)*S;
fprintf('Ledoit-Wolf shrinkage intensity: %.3f\n', shrinkage);

lb = zeros(N,1); ub = ones(N,1);
Aeq = ones(1,N); beq = 1;
qpopts = optimoptions('quadprog','Display','off');

%% ---- (i) GMV: min w'Sigma w s.t. 1'w=1, 0<=w<=1 ----
w_gmv = quadprog(2*Sigma, zeros(N,1), [], [], Aeq, beq, lb, ub, [], qpopts);

%% ---- (ii) Tangency: max w'mu/sqrt(w'Sigma w) s.t. 1'w=1, 0<=w<=1 ----
% Standard reduction (valid since some mu_i>0): min y'Sigma y s.t. mu'y=1, y>=0; w=y/sum(y)
y0 = ones(N,1)/N;
y = quadprog(2*Sigma, zeros(N,1), [], [], mu', 1, zeros(N,1), [], y0, qpopts);
w_tan = y / sum(y);

%% ---- (iii) Risk Parity: log-ERC, equal budget b_i=1/N ----
b = ones(N,1)/N;
ircObj = @(w) logERCObjective(w, Sigma, b);
fmopts = optimoptions('fmincon','Display','off','MaxFunctionEvaluations',2e5,'MaxIterations',5e3,'Algorithm','sqp');
w0 = ones(N,1)/N;
w_rp = fmincon(ircObj, w0, [], [], Aeq, beq, lb, ub, [], fmopts);

%% ---- (iv) CVaR (Rockafellar-Uryasev), alpha=0.95, historical scenarios ----
alpha = 0.95;
% Variables: [w (N); zeta (1); u (Tn)]
nvar = N + 1 + Tn;
f = [zeros(N,1); 1; ones(Tn,1)/((1-alpha)*Tn)];
% u_m >= -R(m,:)*w - zeta  =>  -R(m,:)*w - zeta - u_m <= 0
Acvar = [-R, -ones(Tn,1), -eye(Tn)];
bcvar = zeros(Tn,1);
Aeq2 = [ones(1,N), 0, zeros(1,Tn)];
beq2 = 1;
lb2 = [lb; -Inf; zeros(Tn,1)];
ub2 = [ub;  Inf; Inf(Tn,1)];
lpopts = optimoptions('linprog','Display','off');
xcvar = linprog(f, Acvar, bcvar, Aeq2, beq2, lb2, ub2, lpopts);
w_cvar = xcvar(1:N);

%% ---- (v) Turnover-regularized mean-variance (static proxy, per user 2026-08-25) ----
% eq. (12) is inherently a rolling/path-dependent construction (needs
% w_{tau-1} at each rebalance date, plus author-chosen lambda_mu/lambda_TO,
% none of which are specified in the repo). Static one-shot proxy agreed
% with the user: penalize L1 deviation from the equal-weight portfolio
% instead of a real previous-period weight, lambda_mu = lambda_TO = 1.
lambda_mu = 1; lambda_TO = 1;
w_eq = ones(N,1)/N;
% variables x = [w (N); t (N)], t_i >= |w_i - w_eq_i|
Hto = blkdiag(2*Sigma, zeros(N));
fto = [-lambda_mu*mu; lambda_TO*ones(N,1)];
Ato = [eye(N), -eye(N); -eye(N), -eye(N)];
bto = [w_eq; -w_eq];
Aeqto = [ones(1,N), zeros(1,N)];
beqto = 1;
lbto = [lb; zeros(N,1)]; ubto = [ub; Inf(N,1)];
xto = quadprog(Hto, fto, Ato, bto, Aeqto, beqto, lbto, ubto, [], qpopts);
w_to = xto(1:N);

%% ---- (vi) BL_Tangency (equal-weight market, no active views, per user 2026-08-25) ----
% No views specified anywhere in the repo/manuscript text -- agreed default
% with the user: w^mkt = 1/N, P/Q/Omega empty, so the BL posterior mean
% collapses to the equilibrium prior pi = delta*Sigma*w^mkt (eq. 13 with no
% view terms), and BL_Tangency = Tangency (eq. 9) applied to pi instead of
% the sample mean mu. delta (market risk aversion) backed out from the
% market portfolio's own implied Sharpe ratio: delta = (w^mkt'mu)/(w^mkt'Sigma*w^mkt).
w_mkt = ones(N,1)/N;
delta = (w_mkt'*mu) / (w_mkt'*Sigma*w_mkt);
pi_bl = delta * Sigma * w_mkt;
y0 = ones(N,1)/N;
y_bl = quadprog(2*Sigma, zeros(N,1), [], [], pi_bl', 1, zeros(N,1), [], y0, qpopts);
w_bl = y_bl / sum(y_bl);

%% ---- Assemble & report ----
W = table(string(names)', w_gmv, w_tan, w_rp, w_cvar, w_to, w_bl, ...
    'VariableNames', {'Series','GMV','Tangency','RiskParity','CVaR','Turnoverpen','BL_Tangency'});
disp(W);
writetable(W, 'tab_portfolio_weights_18.csv');

portRet  = @(w) mean(R*w)*12;
portVol  = @(w) std(R*w)*sqrt(12);
portSharpe = @(w) portRet(w)/portVol(w);
names6 = ["GMV";"Tangency";"RiskParity";"CVaR";"Turnoverpen";"BL_Tangency"];
Wmat = [w_gmv, w_tan, w_rp, w_cvar, w_to, w_bl];
summ = table(names6, ...
    arrayfun(@(k) portRet(Wmat(:,k)), (1:6)'), ...
    arrayfun(@(k) portVol(Wmat(:,k)), (1:6)'), ...
    arrayfun(@(k) portSharpe(Wmat(:,k)), (1:6)'), ...
    'VariableNames', {'Portfolio','AnnReturn','AnnVol','Sharpe'});
disp(summ);
writetable(summ, 'tab_portfolio_summary_18.csv');

%% ---- helper ----
function val = logERCObjective(w, Sigma, b)
    sp = sqrt(w'*Sigma*w);
    RC = w .* (Sigma*w) / sp;
    RC = max(RC, 1e-12);   % guard log() domain during line search
    N = numel(w);
    val = 0;
    for i = 1:N-1
        for j = i+1:N
            val = val + (log(RC(i)) - log(b(i)) - log(RC(j)) + log(b(j)))^2;
        end
    end
end
