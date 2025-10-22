function res = fit_tb(y, kind, dist)
% Toolbox GARCH/GJR/EGARCH wrapper
switch lower(kind)
    case 'garch',  M = garch(1,1);
    case 'gjr',    M = gjr(1,1);
    case 'egarch', M = egarch(1,1);
    otherwise, error('kind');
end
M.Distribution = dist;
[Est,~,logL] = estimate(M, y, 'Display','off');
[v,z] = infer(Est, y);

[pvec, names] = flattenParams(Est);
k = numel(pvec); n = numel(z);
[AIC,BIC] = ic(logL,k,n);

res = struct('name',upper(kind),'params',pvec,'paramNames',{names}, ...
             'sigma2',v,'stdz',z,'logL',logL,'AIC',AIC,'BIC',BIC, ...
             'diag', diagz(z), 'd', NaN, 'tstat_d', NaN, ...
             'paramTable', table(names(:), pvec(:), nan(size(pvec(:))), ...
                                 'VariableNames',{'param','est','se'}));
end

function res = fit_fig(y, fg)
T = numel(y);
if isempty(fg.Kmax), fg.Kmax = min(1500, max(300, T-1)); end
lb = [1e-12, 0, 0, 1e-6];  ub = [10*var(y), 0.999, 0.999, 0.999];
p0 = [0.05*var(y), 0.2, 0.6, 0.3];
opts = optimoptions('fmincon','Display','off','MaxIterations',fg.maxit,'MaxFunctionEvaluations',fg.maxfev);
obj = @(p) figarch_nll_gauss(y, p, fg.Kmax, fg.burn);

have_fmincon = ~isempty(which('fmincon'));
if have_fmincon
    [pHat, nll, ~, ~, ~, ~, H] = fmincon(obj, p0, [],[],[],[], lb, ub, [], opts);
    se = sqrt(diag(pinv(H)));
else
    penObj = @(x) obj(softbox(x,lb,ub)); x0 = invsoftbox(p0,lb,ub);
    xhat = fminsearch(penObj, x0, optimset('Display','off','MaxIter',fg.maxit,'MaxFunEvals',fg.maxfev));
    pHat = softbox(xhat,lb,ub); se = nan(4,1);
    nll = obj(pHat);
end

[~, v, eps] = figarch_nll_gauss(y, pHat, fg.Kmax, fg.burn);
z = eps ./ sqrt(v);
logL = -nll; k = numel(pHat); n = numel(z);
[AIC,BIC] = ic(logL,k,n);
tstat_d = pHat(4) ./ se(4);

paramNames = ["omega","phi","beta","d"]';
res = struct('name','FIGARCH','params',pHat,'paramNames',{paramNames}, ...
    'sigma2',v,'stdz',z,'logL',logL,'AIC',AIC,'BIC',BIC, ...
    'diag',diagz(z),'d',pHat(4),'tstat_d',tstat_d, ...
    'paramTable', table(paramNames, pHat(:), se(:), 'VariableNames',{'param','est','se'}));
end

function [AIC,BIC] = ic(logL,k,n), AIC = -2*logL+2*k; BIC = -2*logL+k*log(n); end

function D = diagz(z)
try, [~,p1]=lbqtest(z,'Lags',12); [~,p2]=lbqtest(z.^2,'Lags',12); catch, p1=NaN; p2=NaN; end
try, [~,p3]=archtest(z,'Lags',12); catch, p3=NaN; end
D = struct('LBz_p',p1,'LBz2_p',p2,'ARCHLM_p',p3);
end

function [pvec, names] = flattenParams(Est)
% extract scalar numeric fields in a stable order
fn = fieldnames(Est); pvec = []; names = strings(0,1);
for k=1:numel(fn)
    v = Est.(fn{k});
    if isnumeric(v) && isscalar(v) && isfinite(v)
        pvec(end+1,1) = v; names(end+1,1) = string(fn{k}); %#ok<AGROW>
    end
end
end

function p = softbox(x, lb, ub), p = lb + (ub-lb)./(1+exp(-x)); end
function x = invsoftbox(p, lb, ub), z=(p-lb)./(ub-lb); z = min(max(z,1e-10),1-1e-10); x = -log(1./z - 1); end
