function OUT = fit_auto_vol_models(R, names)
% FIT_AUTO_VOL_MODELS  Auto-select volatility models per series.
% Models tried per index: GARCH(1,1), GJR(1,1), EGARCH(1,1) (Econometrics Toolbox),
% and FIGARCH(1,d,1) via Gaussian QML (requires figarch_nll_gauss.m on path).
%
% INPUTS
%   R     : T x N matrix of returns (will be demeaned if cfg.demean = true)
%   names : N x 1 string/cellstr of series names (optional)
%
% OUTPUT (scalar struct)
%   OUT.choice{i}   : chosen model name for series i
%   OUT.stats{i}    : table [logL, AIC, BIC] for chosen model
%   OUT.params{i}   : table of parameters (name, est, se) for chosen model
%   OUT.all{i}      : struct array of all models tried for series i
%   OUT.sigma2(:,i) : conditional variance of chosen model (aligned to R)
%   OUT.stdz(:,i)   : standardized residuals of chosen model (aligned to R)
%
% Notes:
%  - Selection metric: BIC with guardrails:
%       * Penalize models that fail residual diagnostics (LB on z, z^2; ARCH-LM).
%       * Demote FIGARCH if fractional parameter d is not significant (|t| < 1.96).
%  - If you do not have the Econometrics Toolbox, the code will still fit FIGARCH.
%
% Example:
%   names = ["SAfw","Ausfw","Argfw","chifw","SP500","MSCIEM","Bond","Gold","Liv-ex100"]';
%   OUT = fit_auto_vol_models(R, names);

% ---------- inputs ----------
if nargin < 2 || isempty(names)
    names = "Idx"+(1:size(R,2));
end
if iscell(names), names = string(names); end

[T,N] = size(R);

% ---------- configuration ----------
cfg.demean  = true;
cfg.dist    = 'Gaussian';       % toolbox models: 'Gaussian' or 't'
cfg.lb_sig  = 0.05;             % p-value threshold for diag tests
cfg.selectBy= 'BIC';            % 'BIC' or 'AIC'
cfg.figarch.maxit = 2000;
cfg.figarch.maxfev= 2e5;
% BUG FIX (2026-10-09, peer-review-flagged): burn was a hardcoded 50
% regardless of T. On the SHORT panel (T~65) this left only ~15
% post-burn-in observations contributing to FIGARCH's logL, while
% GARCH/GJR/EGARCH (toolbox exact likelihood) use the full T -- so
% FIGARCH's BIC penalty k*log(n) was computed on an effective sample
% ~4x smaller than its competitors', invalidating the cross-model BIC
% comparison in Tables 4-6. Scale burn to T (capped at the original 50)
% so the likelihood-contributing sample is comparable across candidates.
cfg.figarch.burn  = min(50, floor(T/4));
cfg.figarch.Kmax  = [];         % auto = min(1500, max(300, T-1))

hasTB = license('test','Econometrics_Toolbox') && ~isempty(which('estimate'));

if cfg.demean
    R = R - mean(R,'omitnan');
end

% ---------- outputs (scalar struct!) ----------
OUT = struct();
OUT.choice = cell(N,1);
OUT.stats  = cell(N,1);
OUT.params = cell(N,1);
OUT.all    = cell(N,1);
OUT.sigma2 = nan(T,N);
OUT.stdz   = nan(T,N);

% ---------- main loop ----------
for i = 1:N
    y = R(:,i);
    good = isfinite(y);
    yt = y(good);

%     if numel(yt) < 100
%         warning('%s: too few observations (%d). Skipping.', string(names(i)), numel(yt));
%         continue;
%     end

    % ---- try models (collect in cell array to avoid struct concat errors) ----
    trialCells = {};

    % BUG FIX (2026-10-19, co-author Faff flagged pre-resubmission): FIGARCH's
    % own logL/AIC/BIC were already made internally consistent (burn-in and
    % parameter-count fixes above), but they were never put on a COMMON
    % evaluation sample with GARCH/GJR/EGARCH's. fit_fig scores its
    % likelihood only over t=burn+1:T (figarch_nll_gauss.m holds sigma2 at
    % the unconditional variance for t=1:burn and excludes those terms from
    % the sum), while fit_tb's toolbox-based logL spans the FULL t=1:T via
    % estimate()/infer(). Comparing BIC across models whose log-likelihoods
    % are sums over different numbers of terms is not a valid comparison
    % regardless of how each model's own n enters its BIC penalty. Fix: pass
    % the SAME burn to fit_tb and have it also report logL/AIC/BIC restricted
    % to t=burn+1:T, so every candidate for a given series is compared on an
    % identical evaluation sample and convention. Diagnostics (stdz-based LB/
    % ARCH-LM tests) are left on the full sample, since those are each
    % model's own internal residual check, not a cross-model comparison.
    cBurn = cfg.figarch.burn;
    if hasTB
        try, trialCells{end+1} = fit_tb(yt,'garch',  cfg.dist, cBurn); catch ME, warning('[%s] GARCH skipped: %s',  string(names(i)), ME.message); end
        try, trialCells{end+1} = fit_tb(yt,'gjr',    cfg.dist, cBurn); catch ME, warning('[%s] GJR skipped: %s',    string(names(i)), ME.message); end
        try, trialCells{end+1} = fit_tb(yt,'egarch', cfg.dist, cBurn); catch ME, warning('[%s] EGARCH skipped: %s', string(names(i)), ME.message); end
    end
    % FIGARCH (QML)
    try
        trialCells{end+1} = fit_fig(yt, cfg.figarch);
    catch ME
        warning('[%s] FIGARCH failed: %s', string(names(i)), ME.message);
    end

    if isempty(trialCells)
        warning('%s: no model estimated.', string(names(i)));
        continue;
    end

    % Convert to consistent struct array
    trials = [trialCells{:}];

    % ---- selection with guardrails ----
    selMetric = upper(cfg.selectBy);
    M = numel(trials);
    val    = arrayfun(@(s) s.(selMetric), trials);
    % diagnostics pass flag
    pass   = arrayfun(@(s) all(([s.diag.LBz_p s.diag.LBz2_p s.diag.ARCHLM_p] >= cfg.lb_sig) ...
                     | isnan([s.diag.LBz_p s.diag.LBz2_p s.diag.ARCHLM_p])), trials);
    penal  = double(~pass) * 2;                 % add small penalty to metric if fails
    excluded = false(M,1);
    for k=1:M
        if strcmpi(trials(k).name,'FIGARCH')
            td = trials(k).tstat_d;
            if ~(isfinite(td) && abs(td) >= 1.96)
                excluded(k) = true;             % guardrail: drop FIGARCH if d insignif.
            end
        end
    end
    valAdj = val + penal;
    valAdj(excluded) = Inf;
    if all(~isfinite(valAdj))
        valAdj = val + penal;                   % guardrail would exclude everything; fall back
    end
    [~, bestIdx] = min(valAdj);
    best = trials(bestIdx);

    % ---- store chosen model ----
    OUT.choice{i} = best.name;

% Coerce possibly empty/non-scalar logL/AIC/BIC to numeric scalars
logL_ = best.logL;  if isempty(logL_) || ~isnumeric(logL_) || ~isscalar(logL_) || ~isfinite(logL_), logL_ = NaN; end
AIC_  = best.AIC;   if isempty(AIC_)  || ~isnumeric(AIC_)  || ~isscalar(AIC_)  || ~isfinite(AIC_),  AIC_  = NaN; end
BIC_  = best.BIC;   if isempty(BIC_)  || ~isnumeric(BIC_)  || ~isscalar(BIC_)  || ~isfinite(BIC_),  BIC_  = NaN; end

OUT.stats{i}  = table(logL_, AIC_, BIC_, 'VariableNames', {'logL','AIC','BIC'});
OUT.params{i} = best.paramTable;

zFull = nan(size(y)); vFull = nan(size(y));
zFull(good) = best.stdz; vFull(good) = best.sigma2;
OUT.stdz(:,i)   = zFull;
OUT.sigma2(:,i) = vFull;
OUT.all{i}      = trials;

fprintf('[%s] Selected: %s | %s = %.2f | d = %.3f (t = %.2f)\n', ...
    string(names(i)), best.name, selMetric, best.(selMetric), best.d, best.tstat_d);
%%%
    OUT.params{i} = best.paramTable;

    zFull = nan(size(y)); vFull = nan(size(y));
    zFull(good) = best.stdz; vFull(good) = best.sigma2;
    OUT.stdz(:,i)  = zFull;
    OUT.sigma2(:,i)= vFull;
    OUT.all{i}     = trials;

    fprintf('[%s] Selected: %s | %s = %.2f | d = %.3f (t = %.2f)\n', ...
        string(names(i)), best.name, selMetric, best.(selMetric), best.d, best.tstat_d);
end
end

% ======================================================================
% ======================== helper subfunctions ==========================
% ======================================================================

function res = fit_tb(y, kind, dist, burn)
% Robust wrapper for Econometrics Toolbox models: GARCH/GJR/EGARCH
if nargin < 4 || isempty(burn), burn = 0; end
k = strtrim(lower(kind));
switch k
    case 'garch',  ctor = @() garch(1,1);
    case 'gjr',    ctor = @() gjr(1,1);
    case 'egarch', ctor = @() egarch(1,1);
    otherwise, error('fit_tb:unknownKind','Unknown volatility model "%s".', kind);
end

try
    M = ctor();
catch ME
    error('fit_tb:noCtor','Constructor for "%s" not available: %s', k, ME.message);
end
if nargin<3 || isempty(dist), dist = 'Gaussian'; end
M.Distribution = dist;

% BUG FIX (2026-10-22, co-author Faff flagged pre-submission): the
% previous version fit M via estimate(M, y, ...) over the FULL sample,
% then only *scored* logL over t=burn+1:T afterward using those
% full-sample-optimal parameters -- i.e. a full-sample MLE rescored on a
% sub-window, not a genuine restricted-sample MLE. Robert Faff flagged
% this directly: "confirm that each candidate maximises the likelihood
% over that common window, rather than having full-sample estimates
% subsequently scored over it." It was not maximizing over the common
% window. This also explains the "some GJR likelihoods remain below
% their nested GARCH counterparts" observation: the nested-model result
% (GJR's logL >= GARCH's logL) only holds when both are each maximized
% over the IDENTICAL sample; scoring non-maximizing parameters over a
% sub-window has no such guarantee. Fix: genuinely re-estimate via
% estimate() using only y(burn+1:end) as the in-sample data, with
% y(1:burn) supplied as presample conditioning (E0/V0) -- mirroring how
% figarch_nll_gauss.m already uses t=1:burn purely to initialize
% FIGARCH's long-memory recursion without including those terms in its
% own likelihood sum.
burn = max(0, min(round(burn), numel(y)-10));
if burn > 0
    e0 = y(1:burn);
    v0 = var(e0,1);
    if ~isfinite(v0) || v0 <= 0, v0 = var(y,1); end
    ySample = y(burn+1:end);
    [Est,EstParamCov,logL] = estimate(M, ySample, 'E0', e0, 'V0', v0, 'Display','off');
else
    [Est,EstParamCov,logL] = estimate(M, y, 'Display','off');
end

% BUG FIX (2026-10-22, found while implementing the fix above): calling
% infer(Est, y) on the FULL series with its own DEFAULT presample (derived
% from y's full-sample statistics) is numerically unstable for a model
% whose parameters were estimated on the restricted post-burn window --
% confirmed directly for this panel (DAX_USD's EGARCH: conditional
% variance exploded to ~1.4e41 under the default presample, vs. a
% well-behaved max of ~0.011 when given the SAME presample used during
% estimation). The mismatch between the training presample (E0/V0, or the
% toolbox's own full-sample default pre-fix) and the full-y default
% presample triggers runaway recursion for high-persistence EGARCH fits.
% Fix: infer the post-burn window using the identical E0/V0 presample used
% for estimation (well-behaved by construction), and fill the burn-in
% window itself with the model's unconditional variance -- the same
% treatment figarch_nll_gauss.m already gives FIGARCH's own burn-in
% (held at the unconditional level, never dynamically inferred there).
if burn > 0
    vTail = infer(Est, ySample, 'E0', e0, 'V0', v0);
    try
        vHead0 = Est.UnconditionalVariance;
    catch
        vHead0 = NaN;
    end
    if isempty(vHead0) || ~isfinite(vHead0) || vHead0 <= 0, vHead0 = v0; end
    v = [repmat(vHead0, burn, 1); vTail];
else
    v = infer(Est, y);      % full-length conditional variance for downstream use
end
z = y ./ sqrt(v);           % standardized residuals computed directly

[pvec, names] = flattenParams(Est);
% BUG FIX (2026-10-09, peer-review-flagged): kpars used to come from
% flattenParams, which only picked up isnumeric&&isscalar fields of Est --
% for garch/gjr/egarch that means {P,Q,Constant,Offset,UnconditionalVariance},
% a fixed set of 5 fields for EVERY model type, since the real fitted
% coefficients (ARCH/GARCH/Leverage) live in cell arrays and were silently
% skipped. This made kpars=5 identically for GARCH, GJR, and EGARCH
% regardless of each model's true parameter count, corrupting every BIC in
% Tables 4-6 (this is exactly what the peer review's "why do GARCH, GJR,
% and EGARCH all imply the same parameter count" question caught). Fix:
% kpars is the number of ACTUALLY ESTIMATED free parameters, i.e. the size
% of estimate()'s own parameter-covariance matrix (GARCH=3, GJR=EGARCH=4
% for order (1,1), confirmed directly against the toolbox).
kpars = size(EstParamCov,1);

% n matches FIGARCH's n_used exactly: both are T-burn, the number of
% observations actually contributing to the (now genuinely maximized)
% likelihood sum.
n = numel(y) - burn;
[AIC,BIC] = ic(logL, kpars, n);

res = struct('name', upper(kind), ...
             'params', pvec, 'paramNames', {names}, ...
             'sigma2', v, 'stdz', z, ...
             'logL', logL, 'AIC', AIC, 'BIC', BIC, ...
             'diag', diagz(z), ...
             'd', NaN, 'tstat_d', NaN, ...
             'paramTable', table(names(:), pvec(:), nan(size(pvec(:))), ...
                                 'VariableNames', {'param','est','se'}));
end

function res = fit_fig(y, fg)
% FIGARCH(1,d,1) via Gaussian QML objective figarch_nll_gauss
T = numel(y);
if isempty(fg.Kmax), fg.Kmax = min(1500, max(300, T-1)); end
lb = [1e-12, 0, 0, 1e-6];                 % [omega, phi, beta, d]
ub = [10*var(y), 0.999, 0.999, 0.999];
p0 = [0.05*var(y), 0.2, 0.6, 0.3];

have_fmincon = ~isempty(which('fmincon'));
opts = optimoptions('fmincon','Display','off','MaxIterations',fg.maxit,'MaxFunctionEvaluations',fg.maxfev);
obj  = @(p) figarch_nll_gauss(y, p, fg.Kmax, fg.burn);

if have_fmincon
    [pHat, nll, ~, ~, ~, ~, H] = fmincon(obj, p0, [],[],[],[], lb, ub, [], opts);
    se = sqrt(diag(pinv(H)));              % Hessian-based SEs
else
    penObj = @(x) obj(softbox(x,lb,ub)); x0 = invsoftbox(p0,lb,ub);
    xhat   = fminsearch(penObj, x0, optimset('Display','off','MaxIter',fg.maxit,'MaxFunEvals',fg.maxfev));
    pHat   = softbox(xhat,lb,ub); se = nan(4,1); nll = obj(pHat);
end

[~, v, eps, n_used] = figarch_nll_gauss(y, pHat, fg.Kmax, fg.burn);
z = eps ./ sqrt(v);

logL = -nll; kpars = numel(pHat);
n = n_used; % BUG FIX (2026-09-03): was numel(z) = T, the full series
            % including the un-fit burn-in period; use the actual number
            % of observations contributing to logL (see figarch_nll_gauss.m).
[AIC,BIC] = ic(logL, kpars, n);
tstat_d   = pHat(4) ./ se(4);

paramNames = ["omega","phi","beta","d"]';
res = struct('name','FIGARCH', ...
             'params', pHat, 'paramNames', {paramNames}, ...
             'sigma2', v, 'stdz', z, ...
             'logL', logL, 'AIC', AIC, 'BIC', BIC, ...
             'diag', diagz(z), ...
             'd', pHat(4), 'tstat_d', tstat_d, ...
             'paramTable', table(paramNames, pHat(:), se(:), 'VariableNames', {'param','est','se'}));
end

function [AIC,BIC] = ic(logL,k,n)
AIC = -2*logL + 2*k;
BIC = -2*logL + k*log(n);
end

function D = diagz(z)
% Basic residual diagnostics: Ljung-Box on z and z^2, ARCH-LM
try, [~,p1] = lbqtest(z,'Lags',12);  catch, p1 = NaN; end
try, [~,p2] = lbqtest(z.^2,'Lags',12); catch, p2 = NaN; end
try, [~,p3] = archtest(z,'Lags',12);  catch, p3 = NaN; end
D = struct('LBz_p',p1,'LBz2_p',p2,'ARCHLM_p',p3);
end

function [pvec, names] = flattenParams(EstMdl)
% Extract the ACTUALLY ESTIMATED coefficients of a garch/gjr/egarch object.
% BUG FIX (2026-10-09, peer-review-flagged): the previous version grabbed
% any isnumeric&&isscalar field, which picks up {P,Q,Constant,Offset,
% UnconditionalVariance} -- model ORDERS and a DERIVED quantity, not fitted
% parameters -- while silently skipping ARCH/GARCH/Leverage, which the
% toolbox stores as cell arrays (one cell per lag) and are the real
% estimated coefficients. Constant is kept; P, Q, Offset (fixed at 0 by
% default), and UnconditionalVariance (a function of the other parameters,
% not a free parameter) are excluded.
pvec = EstMdl.Constant; names = "Constant";
if isprop(EstMdl,'ARCH') || isfield(EstMdl,'ARCH')
    for j = 1:numel(EstMdl.ARCH)
        pvec(end+1,1) = EstMdl.ARCH{j}; names(end+1,1) = "ARCH"+j; %#ok<AGROW>
    end
end
if isprop(EstMdl,'Leverage') || isfield(EstMdl,'Leverage')
    for j = 1:numel(EstMdl.Leverage)
        pvec(end+1,1) = EstMdl.Leverage{j}; names(end+1,1) = "Leverage"+j; %#ok<AGROW>
    end
end
if isprop(EstMdl,'GARCH') || isfield(EstMdl,'GARCH')
    for j = 1:numel(EstMdl.GARCH)
        pvec(end+1,1) = EstMdl.GARCH{j}; names(end+1,1) = "GARCH"+j; %#ok<AGROW>
    end
end
pvec = pvec(:); names = names(:);
end

function p = softbox(x, lb, ub)
% Map R^n -> [lb,ub] via logistic
p = lb + (ub-lb) ./ (1 + exp(-x));
end

function x = invsoftbox(p, lb, ub)
% Inverse mapping [lb,ub] -> R^n
z = (p - lb) ./ (ub - lb);
z = min(max(z, 1e-10), 1-1e-10);
x = -log(1./z - 1);
end

function x = scalarOrNaN(v)
    if isempty(v) || ~isscalar(v) || ~isnumeric(v)
        x = NaN;
    else
        x = v;
    end
end

function x = safeScalar(v)
% Return a numeric scalar; otherwise NaN (avoids table varname mismatch)
    if isempty(v) || ~isnumeric(v) || ~isscalar(v) || ~isfinite(v)
        x = NaN;
    else
        x = v;
    end
end