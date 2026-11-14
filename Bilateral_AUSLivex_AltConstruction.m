%% Bilateral_AUSLivex_AltConstruction.m (2026-11-06, ARS panel S1)
% -------------------------------------------------------------------------
% Devil's Advocate finding (MAJOR #4): the paper's one significant
% dependence result (the bilateral Australia-vs-Liv-ex DCC/ADCC
% specification, Section 5.6, on raw hybrid-index returns) is never
% re-tested under the alternative Australian index constructions (pooled
% time-dummy hedonic, chained hedonic) already used for every other
% headline Australian claim in Appendix B.3. This script re-estimates the
% identical bilateral DCC/ADCC specification on raw AUS_wine_USD returns
% from the two alternative-construction master panels.
% -------------------------------------------------------------------------
clear; clc;

files = struct('tag', {'Hybrid (original)','PooledHedonic','ChainedHedonic'}, ...
    'path', {'MasterData_USD_2019_2024_4country.mat', ...
             'MasterData_USD_2019_2024_4country_HedonicAUS.mat', ...
             'MasterData_USD_2019_2024_4country_ChainedHedonicAUS.mat'});

opts = optimoptions(@fmincon, 'Display','off', 'Algorithm','sqp', ...
    'TolFun',1e-10, 'TolCon',1e-10, 'MaxFunctionEvaluations',5e4, 'MaxIterations',5e3);

results = table('Size',[0 6], 'VariableTypes', ["string","double","double","double","double","double"], ...
    'VariableNames', ["Construction","DCC_a","DCC_a_t","ADCC_g","ADCC_g_t","StdAUS"]);

for k = 1:numel(files)
    S = load(files(k).path, 'M');
    T = S.M;
    r_aus = diff(log(T.AUS_wine_USD));
    r_liv = diff(log(T.LivEx100_USD));

    % Raw-series GARCH-family filter (no guardrail bypass -- same pipeline step
    % the original Section 5.6 bilateral check applies before DCC/ADCC).
    outAus = fit_auto_vol_models(r_aus, "AUS_wine_USD");
    outLiv = fit_auto_vol_models(r_liv, "LivEx100_USD");
    z_aus = outAus.stdz; z_liv = outLiv.stdz;
    n = min(numel(z_aus), numel(z_liv));
    Z = [z_aus(end-n+1:end), z_liv(end-n+1:end)];
    Z = Z - mean(Z,1,'omitnan');

    Qbar = cov(Z, 'partialrows');
    Sbar = cov(Z .* (Z < 0), 'partialrows');

    x0 = [0.05; 0.90]; lb=[0;0]; ub=[1;1]; Adcc=[1 1]; bdcc=0.999;
    [th_dcc, ~, ~, ~, ~, ~, H_dcc] = fmincon(@(x) localDCCNLL(x,Z,Qbar), x0, Adcc, bdcc, [],[],lb,ub,[],opts);
    se_dcc = sqrt(diag(inv(H_dcc))); t_dcc = th_dcc ./ se_dcc;

    x0a = [0.05; 0.90; 0.05]; lba=[0;0;0]; uba=[1;1;1]; Aadcc=[1 1 0.5]; badcc=0.999;
    [th_adcc, ~, ~, ~, ~, ~, H_adcc] = fmincon(@(x) localADCCNLL(x,Z,Qbar,Sbar), x0a, Aadcc, badcc, [],[],lba,uba,[],opts);
    se_adcc = sqrt(diag(inv(H_adcc))); t_adcc = th_adcc ./ se_adcc;

    fprintf('[%s] AUS model=%s, std(r_aus)=%.4f | DCC a=%.4f (t=%.2f) | ADCC g=%.4f (t=%.2f)\n', ...
        files(k).tag, outAus.choice{1}, std(r_aus), th_dcc(1), t_dcc(1), th_adcc(3), t_adcc(3));

    results = [results; {files(k).tag, th_dcc(1), t_dcc(1), th_adcc(3), t_adcc(3), std(r_aus)}]; %#ok<AGROW>
end

disp(results);
writetable(results, 'tab_bilateral_altconstruction.csv');
fprintf('Saved tab_bilateral_altconstruction.csv\n');

function nll = localDCCNLL(theta, u, Qbar)
    [Tn,~] = size(u); a = theta(1); b = theta(2); Q = Qbar; nll = 0;
    for t = 1:Tn
        Rt = diag(1./sqrt(diag(Q))) * Q * diag(1./sqrt(diag(Q))); Rt = (Rt+Rt')/2;
        ut = u(t,:)'; detR = det(Rt);
        if detR <= 0 || any(~isfinite(ut)), continue; end
        nll = nll + 0.5*(log(detR) + ut'*(Rt\ut) - ut'*ut);
        Q = (1-a-b)*Qbar + a*(ut*ut') + b*Q;
    end
end
function nll = localADCCNLL(theta, u, Qbar, Sbar)
    [Tn,~] = size(u); a = theta(1); b = theta(2); g = theta(3); Q = Qbar; nll = 0;
    for t = 1:Tn
        Rt = diag(1./sqrt(diag(Q))) * Q * diag(1./sqrt(diag(Q))); Rt = (Rt+Rt')/2;
        ut = u(t,:)'; detR = det(Rt);
        if detR <= 0 || any(~isfinite(ut)), continue; end
        nll = nll + 0.5*(log(detR) + ut'*(Rt\ut) - ut'*ut);
        neg = ut .* (ut<0);
        Q = (Qbar - a*Qbar - b*Qbar - g*Sbar) + a*(ut*ut') + b*Q + g*(neg*neg');
    end
end
