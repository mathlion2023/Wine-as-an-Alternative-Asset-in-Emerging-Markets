%% Unsmoothing_Robustness_4country.m (peer-review fix, R5; extended R3 2026-11-06)
% Devil's Advocate / Perspective reviewer finding: the paper names the
% appraisal-index return-smoothing alternative explanation for its
% panel-wide null DCC/ADCC result (Geltner, 1991; Section 5.3) but never
% tests it. This script applies Geltner's own AR(1) unsmoothing correction
% to the two richest, least-interpolated wine series (Australia 0%
% interpolated, South Africa 27% interpolated) and re-estimates the
% bilateral DCC/ADCC against Liv-ex 100 on the desmoothed series, to check
% whether the null result survives.
%
% 2026-11-06 ARS panel (Perspective W1, Domain W4, Devil's Advocate M5):
% running this check only on AUS/SA tests the smoothing hypothesis on the
% two series LEAST exposed to the interpolation mechanism it is meant to
% probe (0%/27% interpolated) while leaving Argentina (44%) and Chile
% (13%) -- the series actually built by linear interpolation of
% zero-trade months -- untested. Extended below to cover all four
% countries.
%
% Method (Geltner, 1991): assume the observed (appraisal/thin-trading)
% return follows r_t^obs = (1-phi)*r_t^true + phi*r_{t-1}^obs, i.e. the
% observed series is itself a first-order smoothed version of the true
% return. phi is estimated as the first-order autocorrelation of r^obs.
% The desmoothed series is then r_t^true = (r_t^obs - phi*r_{t-1}^obs)/(1-phi).
clear; clc;

T = readtable('MasterData_USD_2019_2024_4country_LogReturns.csv');
names = T.Properties.VariableNames;
dateCol = names{1};

getCol = @(nm) T.(nm);
r_aus = getCol('AUS_wine_USD');
r_sa  = getCol('SA_wine_USD');
r_arg = getCol('ARG_wine_USD');
r_chl = getCol('CHL_wine_USD');
r_liv = getCol('LivEx100_USD');

targets = struct('name', {'AUS_wine_USD','SA_wine_USD','ARG_wine_USD','CHL_wine_USD'}, ...
    'r', {r_aus, r_sa, r_arg, r_chl}, 'interp_pct', {0, 27, 44, 13});

results = table('Size',[0 8], 'VariableTypes', ["string","double","double","double","double","double","double","double"], ...
    'VariableNames', ["Series","phi_AR1","StdObs","StdDesmoothed","DCC_a","DCC_a_t","ADCC_g","ADCC_g_t"]);

opts = optimoptions(@fmincon, 'Display','off', 'Algorithm','sqp', ...
    'TolFun',1e-10, 'TolCon',1e-10, 'MaxFunctionEvaluations',5e4, 'MaxIterations',5e3);

for k = 1:numel(targets)
    name = targets(k).name;
    r_obs = targets(k).r;

    % Step 1: estimate phi as the AR(1) coefficient of the observed series
    y  = r_obs(2:end);
    x  = r_obs(1:end-1);
    phi = (x - mean(x))' * (y - mean(y)) / sum((x-mean(x)).^2);
    phi = max(min(phi, 0.95), -0.95);  % keep away from the unit boundary

    % Step 2: Geltner (1991) AR(1) unsmoothing
    r_true = nan(size(r_obs));
    r_true(1) = r_obs(1);
    for t = 2:numel(r_obs)
        r_true(t) = (r_obs(t) - phi*r_obs(t-1)) / (1 - phi);
    end

    fprintf('--- %s --- phi=%.4f | std(obs)=%.4f -> std(desmoothed)=%.4f\n', ...
        name, phi, std(r_obs), std(r_true));

    % Step 3: fit a GARCH-family model to the desmoothed series (same
    % candidate set and guardrail as the main pipeline) to get fresh
    % standardized residuals.
    outSel = fit_auto_vol_models(r_true, string(name));
    z_wine = outSel.stdz;
    fprintf('    Desmoothed %s selected model: %s\n', name, outSel.choice{1});

    % Step 4: fit the same for Liv-ex 100 raw returns (independent of the
    % wine-side desmoothing) so both series are on a like-for-like,
    % GARCH-filtered basis before DCC/ADCC.
    outLiv = fit_auto_vol_models(r_liv, "LivEx100_USD");
    z_liv  = outLiv.stdz;

    n = min(numel(z_wine), numel(z_liv));
    Z = [z_wine(end-n+1:end), z_liv(end-n+1:end)];
    Z = Z - mean(Z,1,'omitnan');

    Qbar = cov(Z, 'partialrows');
    Sbar = cov(Z .* (Z < 0), 'partialrows');

    % DCC(1,1)
    x0 = [0.05; 0.90]; lb=[0;0]; ub=[1;1]; Adcc=[1 1]; bdcc=0.999;
    [th_dcc, ~, ~, ~, ~, ~, H_dcc] = fmincon(@(x) local_dccNLL(x,Z,Qbar), x0, Adcc, bdcc, [],[],lb,ub,[],opts);
    se_dcc = sqrt(diag(inv(H_dcc)));
    t_dcc = th_dcc ./ se_dcc;

    % ADCC(1,1)
    x0a = [0.05; 0.90; 0.05]; lba=[0;0;0]; uba=[1;1;1]; Aadcc=[1 1 0.5]; badcc=0.999;
    [th_adcc, ~, ~, ~, ~, ~, H_adcc] = fmincon(@(x) local_adccNLL(x,Z,Qbar,Sbar), x0a, Aadcc, badcc, [],[],lba,uba,[],opts);
    se_adcc = sqrt(diag(inv(H_adcc)));
    t_adcc = th_adcc ./ se_adcc;

    fprintf('    Desmoothed DCC:  a=%.4f (t=%.2f), b=%.4f (t=%.2f)\n', th_dcc(1), t_dcc(1), th_dcc(2), t_dcc(2));
    fprintf('    Desmoothed ADCC: a=%.4f (t=%.2f), b=%.4f (t=%.2f), g=%.4f (t=%.2f)\n', ...
        th_adcc(1), t_adcc(1), th_adcc(2), t_adcc(2), th_adcc(3), t_adcc(3));

    results = [results; {name, phi, std(r_obs), std(r_true), th_dcc(1), t_dcc(1), th_adcc(3), t_adcc(3)}];
end

disp(results);
writetable(results, 'tab_unsmoothing_robustness_4country.csv');
fprintf('Saved tab_unsmoothing_robustness_4country.csv\n');

%% ===================== local helper functions ==========================
function nll = local_dccNLL(theta, u, Qbar)
    [T,N] = size(u);
    a = theta(1); b = theta(2);
    Q = Qbar; nll = 0;
    for t = 1:T
        Rt = diag(1./sqrt(diag(Q))) * Q * diag(1./sqrt(diag(Q)));
        Rt = (Rt+Rt')/2;
        ut = u(t,:)';
        detR = det(Rt);
        if detR <= 0 || any(~isfinite(ut)), continue; end
        nll = nll + 0.5*(log(detR) + ut'*(Rt\ut) - ut'*ut);
        Q = (1-a-b)*Qbar + a*(ut*ut') + b*Q;
    end
end

function nll = local_adccNLL(theta, u, Qbar, Sbar)
    [T,N] = size(u);
    a = theta(1); b = theta(2); g = theta(3);
    Q = Qbar; nll = 0;
    for t = 1:T
        Rt = diag(1./sqrt(diag(Q))) * Q * diag(1./sqrt(diag(Q)));
        Rt = (Rt+Rt')/2;
        ut = u(t,:)';
        detR = det(Rt);
        if detR <= 0 || any(~isfinite(ut)), continue; end
        nll = nll + 0.5*(log(detR) + ut'*(Rt\ut) - ut'*ut);
        neg = ut .* (ut<0);
        Q = (Qbar - a*Qbar - b*Qbar - g*Sbar) + a*(ut*ut') + b*Q + g*(neg*neg');
    end
end
