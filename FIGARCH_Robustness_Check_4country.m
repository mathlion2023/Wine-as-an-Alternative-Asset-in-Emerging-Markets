%% FIGARCH_Robustness_Check.m
% -------------------------------------------------------------------------
% Diagnostic (2026-09-03), NOT part of the manuscript table pipeline.
%
% Context: fixing the missing log(2*pi) term + BIC sample-size mismatch in
% figarch_nll_gauss.m (see comments there and in fit_auto_vol_models.m)
% changed FIGARCH's Hessian-based t(d) substantially for two series even
% though the point estimate d barely moved:
%   Chile wine (CHL_wine_USD): t(d) 2.26 -> 1.01
%   IPSA equity (IPSA_USD):    t(d) 30.60 -> 0.009 (huge swing)
%   Gold (Gold_USD):           t(d) 28.13 -> 64.67 (strengthened)
% Adding a parameter-independent constant to an objective should NOT move
% its optimum or true curvature, so this swing is suspicious. The likely
% cause: fmincon's default (interior-point) algorithm returns a
% quasi-Newton (BFGS-style) approximation to the Hessian, accumulated
% along its iteration path -- that path, and hence the approximate
% Hessian's quality, CAN depend on the objective's absolute scale even
% though the true Hessian doesn't. This script checks two things
% independently of fmincon's returned Hessian:
%   (1) Multi-start: does d itself land in the same place from different
%       starting points? (checks for a flat/multi-modal likelihood, i.e.
%       weak identification, vs. genuine optimizer-Hessian sensitivity)
%   (2) Independent numerical Hessian: compute se(d) via central finite
%       differences directly on the (now-correct) objective at the
%       converged point, instead of trusting fmincon's internal H.
% -------------------------------------------------------------------------
clear; clc;
load('MasterData_USD_2019_2024_4country.mat', 'M');
names = M.Properties.VariableNames(2:end);
lvl = M{:, names};
R = diff(log(lvl));
R = R - mean(R,'omitnan');

% CORRECTED HEADER (2026-10-10, code-review finding #13): the paragraph that
% used to sit here was copy-pasted verbatim from the LONG-panel sibling
% script (FIGARCH_Robustness_Check.m) and never updated for this
% SHORT/4-country variant -- it claimed "SA_wine_USD no longer exists in
% this panel" and named MERVAL/IBEX as the targets, both of which describe
% the LONG panel, not this one. This script actually loads the SHORT/
% 4-country panel (line 28, which DOES still contain SA_wine_USD) and tests
% targets=["ARG_wine_USD","CAC_USD"] below. Argentina wine is the SHORT
% panel's one guardrail-override case (FIGARCH wins raw BIC but fails the
% |t(d)|>=1.96 significance rule, Table 4); CAC is retained as a stable
% negative control, matching the LONG-panel script's own convention.
targets = ["ARG_wine_USD","CAC_USD"];

Kmax = min(1500, max(300, size(R,1)-1));
% BUG FIX (2026-10-09, peer-review-flagged): burn must scale with T like
% fit_auto_vol_models.m now does, else this robustness check is run on a
% different effective likelihood window than the main pipeline.
burn = min(50, floor(size(R,1)/4));
lb = [1e-12, 0, 0, 1e-6];
p0_base = [NaN, 0.2, 0.6, 0.3]; % omega filled in per series below

% Multiple starting points: perturb phi/beta/d away from the base start,
% and try three different omega scales, to probe for a flat/multi-modal
% surface.
starts = [
    0.2  0.6  0.30;
    0.5  0.3  0.10;
    0.1  0.9  0.50;
    0.6  0.6  0.05;
    0.3  0.1  0.70;
];

fprintf('%-14s %-6s %10s %10s %12s %12s %12s\n', 'Series','Start','omega0','d_hat','t(d) fmincon','t(d) numHess','nll');

results = table('Size',[0 7], 'VariableTypes', ["string","double","double","double","double","double","double"], ...
    'VariableNames', ["Series","StartIdx","omega0","d_hat","tstat_fmincon","tstat_numHess","nll"]);

for s = 1:numel(targets)
    idx = find(strcmp(names, targets(s)));
    y = R(:, idx);
    v0 = var(y,'omitnan');
    ub = [10*v0, 0.999, 0.999, 0.999];

    for k = 1:size(starts,1)
        p0 = [0.05*v0, starts(k,1), starts(k,2), starts(k,3)];
        obj = @(p) figarch_nll_gauss(y, p, Kmax, burn);
        opts = optimoptions('fmincon','Display','off','MaxIterations',2000,'MaxFunctionEvaluations',2e5);
        try
            [pHat, nll, ~, ~, ~, ~, H] = fmincon(obj, p0, [],[],[],[], lb, ub, [], opts);
        catch ME
            fprintf('%-14s start%d FAILED: %s\n', targets(s), k, ME.message);
            continue;
        end
        se_fmincon = sqrt(diag(pinv(H)));
        t_fmincon = pHat(4) / se_fmincon(4);

        % Independent numerical Hessian via central differences on the
        % scalar objective restricted to the converged point, evaluated
        % in the neighborhood of pHat (not fmincon's internal path).
        Hnum = numHessian(obj, pHat, lb, ub);
        se_num = sqrt(diag(pinv(Hnum)));
        t_num = pHat(4) / se_num(4);

        fprintf('%-14s %-6d %10.4g %10.4f %12.3f %12.3f %12.3f\n', ...
            targets(s), k, p0(1), pHat(4), t_fmincon, t_num, nll);

        results = [results; {targets(s), k, p0(1), pHat(4), t_fmincon, t_num, nll}]; %#ok<AGROW>
    end
    fprintf('\n');
end

writetable(results, 'tab_figarch_robustness_4country.csv');
disp('Saved tab_figarch_robustness_4country.csv');

function H = numHessian(f, x, lb, ub)
% Central-difference numerical Hessian of scalar f at x, respecting box
% bounds (step is shrunk if it would cross a bound).
    n = numel(x);
    H = zeros(n);
    h = 1e-4 * max(abs(x), 1e-3);
    % Keep steps inside bounds
    h = min(h, 0.4*min(x-lb, ub-x));
    h = max(h, 1e-8);
    f0 = f(x);
    for i = 1:n
        for j = i:n
            if i==j
                xp = x; xp(i) = xp(i)+h(i);
                xm = x; xm(i) = xm(i)-h(i);
                H(i,i) = (f(xp) - 2*f0 + f(xm)) / h(i)^2;
            else
                xpp = x; xpp(i)=xpp(i)+h(i); xpp(j)=xpp(j)+h(j);
                xpm = x; xpm(i)=xpm(i)+h(i); xpm(j)=xpm(j)-h(j);
                xmp = x; xmp(i)=xmp(i)-h(i); xmp(j)=xmp(j)+h(j);
                xmm = x; xmm(i)=xmm(i)-h(i); xmm(j)=xmm(j)-h(j);
                H(i,j) = (f(xpp)-f(xpm)-f(xmp)+f(xmm)) / (4*h(i)*h(j));
                H(j,i) = H(i,j);
            end
        end
    end
end
