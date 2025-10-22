%% ========= DCC / ADCC QML with SEs, diagnostics, and extras =========
% INPUTS (you have these already from your pipeline)
%   stdResid : T x N matrix of standardized residuals (z_t), mean ≈ 0, var ≈ 1
%   Qbar     : N x N unconditional correlation of stdResid (positive definite)
%   Sbar     : N x N unconditional “negative shock” covariance (for ADCC)
% DEPENDENCIES you already have:
%   dccNegLogLikelihood(theta, stdResid, Qbar)
%   adccNegLogLikelihood(theta, stdResid, Qbar, Sbar)
%   % Ideally, these return [nll, nll_t] with nll_t = per-observation losses (T x 1).
%   % If only nll is returned, robust SEs will be skipped automatically.
clc; clear all;
load('ResidualSTD.mat')
%Precompute unconditional matrices
Qbar = cov(stdResid, 'partialrows');               
Sbar = cov(stdResid .* (stdResid < 0), 'partialrows'); % for asymmetry

[T, N] = size(stdResid);
% Quiet optimizer
opts = optimoptions(@fmincon, 'Display','off', 'Algorithm','sqp', ...
    'TolFun',1e-10, 'TolCon',1e-10, 'MaxFunctionEvaluations',5e4, 'MaxIterations',5e3);

%% -------------------------- DCC(1,1) -----------------------------------
% theta = [a; b], constraints: a>=0, b>=0, a+b <= c
x0_dcc = [0.05; 0.90];          % sensible starting values
lb_dcc = [0; 0];
ub_dcc = [1; 1];
c_dcc  = 0.999;                 % stability margin
A_dcc  = [1, 1]; b_dcc = c_dcc;

dcc_neglog = @(x) dccNegLogWrap(x, stdResid, Qbar);  % safe wrapper
[theta_dcc, nll_dcc, exit_dcc, out_dcc, lambda_dcc, grad_dcc, H_dcc] = ...
    fmincon(dcc_neglog, x0_dcc, A_dcc, b_dcc, [], [], lb_dcc, ub_dcc, [], opts);

dcc = packResults('DCC', theta_dcc, nll_dcc, exit_dcc, out_dcc, H_dcc, grad_dcc, ...
                  @(x) dccNegLogWrap(x, stdResid, Qbar), T);

dcc.persistence = sum(theta_dcc);                   % a + b
dcc.halflife    = halflife_from_ab(dcc.persistence);
dcc.margin      = c_dcc - dcc.persistence;

fprintf('[DCC] a=%.4f (t=%.2f), b=%.4f (t=%.2f) | a+b=%.4f | half-life=%.1f | AIC=%.2f, BIC=%.2f | exit=%d\n', ...
    theta_dcc(1), dcc.t(1), theta_dcc(2), dcc.t(2), dcc.persistence, dcc.halflife, dcc.AIC, dcc.BIC, exit_dcc);

%% -------------------------- ADCC(1,1) ----------------------------------
% theta = [a; b; g], constraints: >=0 and a + b + g/2 <= c  (conservative)
hasSbar=1;

if hasSbar
    x0_adcc = [0.05; 0.90; 0.05];
    lb_adcc = [0; 0; 0];
    ub_adcc = [1; 1; 1];
    c_adcc  = 0.999;
    A_adcc  = [1, 1, 0.5]; b_adcc = c_adcc;

    adcc_neglog = @(x) adccNegLogWrap(x, stdResid, Qbar, Sbar);
    [theta_adcc, nll_adcc, exit_adcc, out_adcc, lambda_adcc, grad_adcc, H_adcc] = ...
        fmincon(adcc_neglog, x0_adcc, A_adcc, b_adcc, [], [], lb_adcc, ub_adcc, [], opts);

    adcc = packResults('ADCC', theta_adcc, nll_adcc, exit_adcc, out_adcc, H_adcc, grad_adcc, ...
                       @(x) adccNegLogWrap(x, stdResid, Qbar, Sbar), T);

    adcc.persistence = theta_adcc(1) + theta_adcc(2) + 0.5*theta_adcc(3);  % a + b + g/2
    adcc.halflife    = halflife_from_ab(theta_adcc(1) + theta_adcc(2));    % base decay via a+b
    adcc.margin      = c_adcc - adcc.persistence;

    fprintf('[ADCC] a=%.4f (t=%.2f), b=%.4f (t=%.2f), g=%.4f (t=%.2f) | a+b+g/2=%.4f | half-life=%.1f | AIC=%.2f, BIC=%.2f | exit=%d\n', ...
        theta_adcc(1), adcc.t(1), theta_adcc(2), adcc.t(2), theta_adcc(3), adcc.t(3), ...
        adcc.persistence, adcc.halflife, adcc.AIC, adcc.BIC, exit_adcc);
else
    warning('Sbar not found. Skipping ADCC estimation.');
    adcc = [];
end

%% ========================== helper functions ===========================

function out = packResults(modelName, theta, nll, exitflag, output, H, grad, nllfun, T)
    % Assemble estimates, compute SEs (Hessian and, if available, robust sandwich),
    % t-stats, p-values, and information criteria.
    k    = numel(theta);
    logL = -nll;

    % Hessian-based (classic) VCOV ≈ inv(H)/T
    vcov_hess = NaN(k);
    se_hess   = NaN(k,1);
    if ~isempty(H) && all(isfinite(H(:))) && rcond(H) > 1e-12
        vcov_hess = (H \ eye(k)) / T;
        se_hess   = sqrt(diag(vcov_hess));
    end

    % Robust sandwich VCOV if per-observation scores are available
    [ll_t, S_t] = tryScores(nllfun, theta);   % ll_t: T×1, S_t: T×k (or [])
    vcov_rob = NaN(k); se_rob = NaN(k,1);
    if ~isempty(S_t) && ~isempty(H) && all(isfinite(H(:))) && rcond(H) > 1e-12
        J = (S_t' * S_t) / T;                 % OPG/T
        Hinv = H \ eye(k);
        vcov_rob = Hinv * J * Hinv / T;       % Sandwich/T
        se_rob   = sqrt(diag(vcov_rob));
    end

    % Prefer robust if available, else Hessian
    se   = se_rob;   vcov = vcov_rob;
    if any(~isfinite(se)), se = se_hess; vcov = vcov_hess; end

    tval = theta ./ se;
    pval = 2 * (1 - normcdf(abs(tval)));

    AIC = -2*logL + 2*k;
    BIC = -2*logL + k*log(T);

    out = struct('model',modelName,'theta',theta(:),'se',se(:),'t',tval(:),'p',pval(:), ...
                 'se_hess',se_hess(:),'vcov',vcov,'logL',logL,'AIC',AIC,'BIC',BIC, ...
                 'exitflag',exitflag,'output',output,'grad',grad);
end

function [ll_t, S_t] = tryScores(nllfun, theta)
% TRYSC0RES  Attempt to obtain per-observation losses and numerical scores.
% OUTPUTS:
%   ll_t : T×1 vector of per-observation losses ([] if not available)
%   S_t  : T×k matrix of per-observation scores wrt theta ([] if not available)
    ll_t = [];
    S_t  = [];

    % Try to get per-obs loss vector from the user's nll function
    try
        % Preferred signature: [nll, ll_t] = nllfun(theta)
        [~, ll_t_try] = nllfun(theta);
        if ~isempty(ll_t_try) && isvector(ll_t_try)
            ll_t = ll_t_try(:);
        end
    catch
        % Function only returns scalar nll → leave ll_t empty
    end

    % If we got ll_t, build numerical scores by central differences
    if ~isempty(ll_t)
        k    = numel(theta);
        Tloc = numel(ll_t);
        S_t  = zeros(Tloc, k);

        eps0 = 1e-6 * max(1, abs(theta)); % scale-aware steps

        for j = 1:k
            thm = theta; thp = theta;
            thm(j) = thm(j) - eps0(j);
            thp(j) = thp(j) + eps0(j);

            lm = []; lp = [];
            try, [~, lm] = nllfun(thm); catch, end
            try, [~, lp] = nllfun(thp); catch, end

            if isempty(lm) || isempty(lp) || ~isvector(lm) || ~isvector(lp)
                S_t = [];   % cannot do robust VCOV → revert to Hessian
                return;
            end

            lm = lm(:); lp = lp(:);
            if numel(lm) ~= Tloc || numel(lp) ~= Tloc
                S_t = [];
                return;
            end

            S_t(:,j) = (lp - lm) ./ (2*eps0(j));
        end
    end
end

function [nll, nll_t] = dccNegLogWrap(theta, Z, Qbar)
% Safe wrapper: always returns scalar nll; returns nll_t if available.
    nll_t = [];
    try
        [nll, nll_t] = dccNegLogLikelihood(theta, Z, Qbar);
    catch
        nll = dccNegLogLikelihood(theta, Z, Qbar);
    end
end

function [nll, nll_t] = adccNegLogWrap(theta, Z, Qbar, Sbar)
% Safe wrapper: always returns scalar nll; returns nll_t if available.
    nll_t = [];
    try
        [nll, nll_t] = adccNegLogLikelihood(theta, Z, Qbar, Sbar);
    catch
        nll = adccNegLogLikelihood(theta, Z, Qbar, Sbar);
    end
end

function hl = halflife_from_ab(persist)
% Half-life (in same periods as your data, e.g., months) for AR(1) decay coeff = persist
    if ~(persist > 0 && persist < 1)
        hl = NaN;
    else
        hl = log(0.5) / log(persist);
    end
end

function nll = dccNegLogLikelihood(par, u, Qbar)
    a = par(1); b = par(2);
    [T, N] = size(u);
    Q = Qbar;
    nll = 0;
    for t = 1:T
        if t>1
            ut = u(t-1,:)';
            Q = (1-a-b)*Qbar + a*(ut*ut') + b*Q;
        end
        D = diag(1./sqrt(diag(Q)));
        R = D*Q*D;
        ut_t = u(t,:)';
        nll = nll + 0.5*( log(det(R)) + ut_t'*(R\ut_t) );
    end
end

function nll = adccNegLogLikelihood(par, u, Qbar, Sbar)
    a = par(1); b = par(2); g = par(3);
    [T,N] = size(u);
    Q = Qbar;
    nll = 0;
    for t = 1:T
        if t>1
            ut = u(t-1,:)';
            u_neg = ut .* (ut<0);
            Q = (1-a-b-g)*Qbar + a*(ut*ut') + b*Q + g*(u_neg*u_neg');
        end
        D = diag(1./sqrt(diag(Q)));
        R = D*Q*D;
        ut_t = u(t,:)';
        nll = nll + 0.5*( log(det(R)) + ut_t'*(R\ut_t) );
    end
end

function Rt = computeDCC(u, Qbar, a, b)
    [T,N] = size(u);
    Q = Qbar;
    Rt = nan(N,N,T);
    for t = 1:T
        if t>1
            ut = u(t-1,:)';
            Q = (1-a-b)*Qbar + a*(ut*ut') + b*Q;
        end
        D = diag(1./sqrt(diag(Q)));
        Rt(:,:,t) = D*Q*D;
    end
end

function Rt = computeADCC(u, Qbar, Sbar, a, b, g)
    [T,N] = size(u);
    Q = Qbar;
    Rt = nan(N,N,T);
    for t = 1:T
        if t>1
            ut = u(t-1,:)';
            u_neg = ut .* (ut<0);
            Q = (1-a-b-g)*Qbar + a*(ut*ut') + b*Q + g*(u_neg*u_neg');
        end
        D = diag(1./sqrt(diag(Q)));
        Rt(:,:,t) = D*Q*D;
    end
end

