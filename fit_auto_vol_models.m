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
cfg.figarch.burn  = 50;
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

    if hasTB
        try, trialCells{end+1} = fit_tb(yt,'garch',  cfg.dist); catch ME, warning('[%s] GARCH skipped: %s',  string(names(i)), ME.message); end
        try, trialCells{end+1} = fit_tb(yt,'gjr',    cfg.dist); catch ME, warning('[%s] GJR skipped: %s',    string(names(i)), ME.message); end
        try, trialCells{end+1} = fit_tb(yt,'egarch', cfg.dist); catch ME, warning('[%s] EGARCH skipped: %s', string(names(i)), ME.message); end
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
    prefer = ones(M,1);
    for k=1:M
        if strcmpi(trials(k).name,'FIGARCH')
            td = trials(k).tstat_d;
            if ~(isfinite(td) && abs(td) >= 1.96)
                prefer(k) = 1.5;                % demote FIGARCH if d insignif.
            end
        end
    end
    [~, bestIdx] = min(val .* prefer + penal);
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

function res = fit_tb(y, kind, dist)
% Robust wrapper for Econometrics Toolbox models: GARCH/GJR/EGARCH
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

[Est,~,logL] = estimate(M, y, 'Display','off');
[v,z] = infer(Est, y);

[pvec, names] = flattenParams(Est);
kpars = numel(pvec); n = numel(z);
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

[~, v, eps] = figarch_nll_gauss(y, pHat, fg.Kmax, fg.burn);
z = eps ./ sqrt(v);

logL = -nll; kpars = numel(pHat); n = numel(z);
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
% Extract scalar numeric fields from toolbox model
fn = fieldnames(EstMdl);
pvec = []; names = strings(0,1);
for j = 1:numel(fn)
    v = EstMdl.(fn{j});
    if isnumeric(v) && isscalar(v) && isfinite(v)
        pvec(end+1,1) = v; %#ok<AGROW>
        names(end+1,1) = string(fn{j}); %#ok<AGROW>
    end
end
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