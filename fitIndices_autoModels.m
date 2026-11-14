%% fitIndices_autoModels.m
% Auto-fit financial econometric models per index and select the best.
% Inputs:
%   R   : T x N matrix of returns (mean-adjusted or raw; we demean by default)
%   names : N x 1 string array of series names
% Options (see opts struct below)
%
% Outputs (struct 'OUT'):
%   OUT.choice{i}        : chosen model name for series i
%   OUT.models{i}        : all model results tried for series i
%   OUT.condVar(:,i)     : conditional variance of chosen model
%   OUT.stdResid(:,i)    : standardized residuals of chosen model
%   OUT.params{i}        : parameter vector of chosen model
%   OUT.stats{i}         : table of [logL, AIC, BIC] for chosen model
%   OUT.diag{i}          : diagnostics for chosen model (LB on z, z^2; ARCH-LM)
%
% USAGE:
%   load your returns (T x N) as R, and series names as indexNames (N x 1):
%   indexNames = ["SAfw","Ausfw","Argfw","chifw","SP500","MSCIEM","Bond","Gold","Liv-ex100"]';
%   OUT = fitIndices_autoModels(R, indexNames);

function OUT = fitIndices_autoModels(R, names)
    if nargin < 2
        names = "Series" + (1:size(R,2));
    end

    % ---------- OPTIONS ----------
    opts.demean = true;
    opts.meanModel = 'AR1';  % 'NONE' | 'AR1' | 'AUTO_AR' (simple auto AR up to 3)
    opts.volModels = {'GARCH','GJR','EGARCH','FIGARCH'}; % order to try
    opts.innovDist = 'Gaussian'; % for toolbox models: 'Gaussian' or 't'
    opts.maxAR = 3;             % if meanModel='AUTO_AR'
    opts.kLjung = 12;           % LB lags
    opts.archLMlags = 12;       % Engle ARCH LM lags
    opts.selectBy = 'BIC';      % 'BIC' or 'AIC'
    opts.figarch.Kmax = [];     % [] => auto set inside
    opts.figarch.burn = 50;
    opts.figarch.maxit = 2000;  % fmincon iterations
    opts.figarch.maxfev = 2e5;

    [T,N] = size(R);
    if opts.demean
        R = R - mean(R,'omitnan');
    end

    hasToolbox = license('test','Econometrics_Toolbox') && ~isempty(which('estimate'));

    % holders
    OUT.choice  = cell(N,1);
    OUT.models  = cell(N,1);
    OUT.condVar = nan(T,N);
    OUT.stdResid= nan(T,N);
    OUT.params  = cell(N,1);
    OUT.stats   = cell(N,1);
    OUT.diag    = cell(N,1);

    for i = 1:N
        y = R(:,i);
        good = isfinite(y);
        yt = y(good);
%         if numel(yt) < 80
%             warning('Series %s: too short (%d obs). Skipping.', names(i), numel(yt));
%             continue;
%         end

        % 1) mean spec
        [muSpec, muParams, eps] = fitMeanSpec(yt, opts);

        % 2) try volatility models
        tried = struct([]);
        for m = 1:numel(opts.volModels)
            mdl = upper(opts.volModels{m});
            try
                switch mdl
                    case 'GARCH'
                        assert(hasToolbox, 'Econometrics Toolbox required for GARCH/GJR/EGARCH.');
                        res = fit_garch_toolbox(eps, 'garch', opts.innovDist);
                    case 'GJR'
                        assert(hasToolbox);
                        res = fit_garch_toolbox(eps, 'gjr', opts.innovDist);
                    case 'EGARCH'
                        assert(hasToolbox);
                        res = fit_garch_toolbox(eps, 'egarch', opts.innovDist);
                    case 'FIGARCH'
                        res = fit_figarch_qml(eps, opts.figarch);
                    otherwise
                        error('Unknown vol model: %s', mdl);
                end
                res.name = mdl;
                res.mean = muSpec;
                res.meanParams = muParams;

                % combine mean+vol back to full length
                z = nan(size(y)); v = nan(size(y));
                z(good) = res.stdz;
                v(good) = res.sigma2;

                % selection metric
                % BUG FIX (2026-10-10, code-review finding #5): BIC sample
                % size used to be numel(res.stdz) = T (the full series) for
                % every model including FIGARCH, which understates FIGARCH's
                % BIC penalty relative to its post-burn-in effective sample
                % (this is the same bug class fixed project-wide on
                % 2026-09-03 in figarch_nll_gauss.m/fit_auto_vol_models.m,
                % but was never propagated to this sibling file). Use the
                % per-model res.n (set to n_used for FIGARCH, T otherwise).
                k = res.nParams + numel(muParams);
                [logL,AIC,BIC] = infoCriteria(res.logL, k, res.n);
                res.logL = logL; res.AIC = AIC; res.BIC = BIC;

                % diagnostics on standardized residuals
                res.diag = basicDiagnostics(res.stdz, opts.kLjung, opts.archLMlags);

                tried = [tried; res]; %#ok<AGROW>
            catch ME
                warning('Series %s: %s failed (%s).', names(i), mdl, ME.message);
            end
        end

        if isempty(tried)
            warning('Series %s: no model converged.', names(i));
            continue;
        end

        % 3) model selection by BIC/AIC; prefer models that pass diagnostics
        selMetric = lower(opts.selectBy);
        metrics = arrayfun(@(s) s.(upper(selMetric)), tried);
        passDiag = arrayfun(@(s) all([s.diag.LBz_p>0.05, s.diag.LBz2_p>0.05, s.diag.ARCHLM_p>0.05] | isnan([s.diag.LBz_p s.diag.LBz2_p s.diag.ARCHLM_p])), tried);
        % penalize failing diagnostics slightly
        penalty = ~passDiag * 2; % add 2 to BIC if fails
        [~,bestIdx] = min(metrics + penalty);
        best = tried(bestIdx);

        % record
        OUT.choice{i}   = best.name;
        OUT.models{i}   = tried;
        OUT.condVar(good,i) = best.sigma2;
        OUT.stdResid(good,i)= best.stdz;
        OUT.params{i}   = struct('mean',best.mean,'meanParams',best.meanParams,'vol',best.name,'volParams',best.params);
        OUT.stats{i}    = table(best.logL, best.AIC, best.BIC, 'VariableNames',{'logL','AIC','BIC'});
        OUT.diag{i}     = best.diag;

        fprintf('[%s] Selected: %s | %s=%0.2f (LBz p=%.3f, LBz^2 p=%.3f, ARCHLM p=%.3f)\n', ...
            names(i), best.name, upper(selMetric), best.(upper(selMetric)), best.diag.LBz_p, best.diag.LBz2_p, best.diag.ARCHLM_p);
    end
end

%% ---------- helpers ----------

function [muSpec, params, eps] = fitMeanSpec(y, opts)
    muSpec = upper(opts.meanModel);
    switch muSpec
        case 'NONE'
            params = [];
            eps = y;
        case 'AR1'
            phi = regress(y(2:end), y(1:end-1));
            c   = mean(y) * (1 - phi); % simple
            u   = y(2:end) - c - phi*y(1:end-1);
            eps = [NaN; u];
            params = [c; phi];
        case 'AUTO_AR'
            maxAR = opts.maxAR;
            bestBIC = inf; bestP = 0; bestU = [];
            for p=0:maxAR
                if p==0
                    u = y - mean(y);
                    k = 1; % mean only
                    logL = -0.5*sum(log(2*pi) + log(var(u)) + (u.^2)/var(u));
                else
                    X = lagmatrix(y,1:p);
                    idx = all(isfinite(X),2);
                    b = [ones(sum(idx),1) X(idx,:)] \ y(idx);
                    u = y(idx) - [ones(sum(idx),1) X(idx,:)]*b;
                    k = numel(b);
                    logL = -0.5*sum(log(2*pi) + log(var(u)) + (u.^2)/var(u));
                    u = align(u, idx, numel(y));
                end
                bic = -2*logL + k*log(sum(isfinite(u)));
                if bic < bestBIC
                    bestBIC = bic; bestP = p; bestU = u;
                end
            end
            eps = bestU;
            params = bestP; % store p only (keep it simple)
            muSpec = sprintf('AR(%d)',bestP);
        otherwise
            error('Unknown mean model: %s', opts.meanModel);
    end
    % replace initial NaN with zero mean innovation
    if isnan(eps(1)), eps(1) = 0; end
end

function u = align(uSub, idx, T)
    u = nan(T,1);
    u(idx) = uSub;
    u(~idx) = NaN;
    if isnan(u(1)), u(1)=0; end
end

function res = fit_garch_toolbox(eps, kind, innovDist)
    switch lower(kind)
        case 'garch'
            M = garch(1,1);
        case 'gjr'
            M = gjr(1,1);
        case 'egarch'
            M = egarch(1,1);
        otherwise, error('Unknown kind');
    end
    M.Distribution = innovDist;
    [EstMdl,EstParamCov,logL] = estimate(M, eps, 'Display','off');   % was 5 outputs -- estimate() on garch/gjr/egarch only returns 3
    v = infer(EstMdl, eps);     % infer() on garch/gjr/egarch returns [V,LogL], not residuals
    z = eps ./ sqrt(v);         % standardized residuals computed directly
    res.params = packParams(EstMdl);
    res.se     = nan(size(res.params));   % toolbox models don't expose SEs here; kept only so struct-concat with fit_figarch_qml's output (which has .se) doesn't fail
    res.sigma2 = v;
    res.stdz   = z;
    res.logL   = logL;
    % BUG FIX (2026-10-10, code-review finding #4): nParams used to be
    % numel(packParams(EstMdl)), but packParams only picked up
    % isnumeric&&isscalar fields of EstMdl -- for garch/gjr/egarch that is
    % the fixed set {P,Q,Constant,Offset,UnconditionalVariance}, giving
    % nParams=5 for EVERY model regardless of true parameter count (the real
    % fitted ARCH/GARCH/Leverage coefficients are stored as cell arrays and
    % were silently skipped). This is the same bug (independently) fixed as
    % flattenParams() in fit_tb.m / fit_auto_vol_models.m on 2026-10-09; this
    % file has its own copy under a different name and was missed by that
    % pass. Fix: use estimate()'s own parameter-covariance matrix size, the
    % number of ACTUALLY ESTIMATED free parameters (GARCH=3, GJR=EGARCH=4).
    res.nParams= size(EstParamCov,1);
    res.n      = numel(z);   % BIC sample size for toolbox models = full T
end

function p = packParams(EstMdl)
    % Extract the ACTUALLY ESTIMATED coefficients of a garch/gjr/egarch
    % object (Constant + ARCH{:} + Leverage{:} + GARCH{:}), not the
    % isnumeric&&isscalar grab-bag this used to be (see BUG FIX note above
    % fit_garch_toolbox, and the identical fix already applied to
    % flattenParams() in fit_tb.m / fit_auto_vol_models.m).
    p = EstMdl.Constant;
    if isprop(EstMdl,'ARCH') || isfield(EstMdl,'ARCH')
        for j = 1:numel(EstMdl.ARCH)
            p(end+1,1) = EstMdl.ARCH{j}; %#ok<AGROW>
        end
    end
    if isprop(EstMdl,'Leverage') || isfield(EstMdl,'Leverage')
        for j = 1:numel(EstMdl.Leverage)
            p(end+1,1) = EstMdl.Leverage{j}; %#ok<AGROW>
        end
    end
    if isprop(EstMdl,'GARCH') || isfield(EstMdl,'GARCH')
        for j = 1:numel(EstMdl.GARCH)
            p(end+1,1) = EstMdl.GARCH{j}; %#ok<AGROW>
        end
    end
    p = p(:);
end

function res = fit_figarch_qml(y, fg)
    % y: residuals after mean model
    T = numel(y);
    if isempty(fg.Kmax)
        fg.Kmax = min(1500, max(300, T-1));
    end
    % BUG FIX (2026-10-10, code-review finding #6): fg.burn used to be a
    % hardcoded 50 regardless of T, reintroducing the burn-in-scaling bug
    % fit_auto_vol_models.m's cfg.figarch.burn = min(50, floor(T/4)) was
    % specifically fixed to avoid on 2026-10-09 (on a short sample, e.g.
    % T~65, this consumed the majority of observations as burn-in, leaving
    % far fewer contributing to FIGARCH's likelihood than the full T used by
    % the competing toolbox models). Scale burn to T here too, capped at the
    % caller's requested value.
    fg.burn = min(fg.burn, floor(T/4));
    lb = [1e-12, 0, 0, 1e-6];     % [omega, phi, beta, d]
    ub = [10*var(y), 0.999, 0.999, 0.999];
    p0 = [0.05*var(y), 0.2, 0.6, 0.3];

    have_fmincon = ~isempty(which('fmincon'));
    opts = optimoptions('fmincon','Display','off','MaxIterations',fg.maxit,'MaxFunctionEvaluations',fg.maxfev);
    obj = @(p) figarch_nll_gauss(y, p, fg.Kmax, fg.burn);

    if have_fmincon
        [pHat, fval, ~, ~, ~, ~, H] = fmincon(obj, p0, [],[],[],[], lb, ub, [], opts);
        % robust QML s.e. (sandwich): we use Hessian-only as an approximation here
        se = sqrt(diag(pinv(H)));
    else
        penObj = @(p) obj(softbox(p,lb,ub));
        pFree0 = invsoftbox(p0,lb,ub);
        pFree  = fminsearch(penObj, pFree0, optimset('Display','off','MaxFunEvals',fg.maxfev,'MaxIter',fg.maxit));
        pHat   = softbox(pFree,lb,ub);
        se = nan(size(pHat));
    end

    [nll, v, epsHat, n_used] = figarch_nll_gauss(y, pHat, fg.Kmax, fg.burn);
    z = epsHat ./ sqrt(v);

    res.params = pHat(:);
    res.se     = se(:);
    res.sigma2 = v;
    res.stdz   = z;
    res.logL   = -nll;               % objective returns NLL
    res.nParams= numel(pHat);
    % BUG FIX (2026-10-10, code-review finding #5): capture figarch_nll_gauss's
    % 4th output (n_used, the number of post-burn-in observations actually
    % contributing to the likelihood) instead of discarding it; the caller
    % now uses res.n (not numel(res.stdz) = T) for FIGARCH's BIC penalty.
    res.n      = n_used;
end

function [logL,AIC,BIC] = infoCriteria(logL, k, n)
    AIC = -2*logL + 2*k;
    BIC = -2*logL + k*log(n);
end

function D = basicDiagnostics(z, kLjung, archLags)
    z = z(:);
    idx = isfinite(z);
    z = z(idx);
    % Ljung-Box on z and z^2 (fallback if toolbox not present)
    try
        [~, p1] = lbqtest(z, 'Lags', kLjung);
        [~, p2] = lbqtest(z.^2, 'Lags', kLjung);
    catch
        p1 = NaN; p2 = NaN;
    end
    % ARCH-LM
    try
        [~, p3] = archtest(z, 'Lags', archLags);
    catch
        p3 = NaN;
    end
    D = struct('LBz_p',p1,'LBz2_p',p2,'ARCHLM_p',p3);
end
