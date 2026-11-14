%% Australia_Only_Robustness.m
% ---------------------------------------------------------------
% Referee 1: Australia-only robustness subsection (volatility,
% dependence, and spillover restricted to Australia vs. Liv-ex 100),
% reported alongside -- not replacing -- the four-country main
% analysis. Uses the same corrected pipeline (guardrail-fixed
% fit_auto_vol_models.m, DCC/ADCC QML, DY connectedness) as the main
% 18-series run, restricted to the two-series subset.
% ---------------------------------------------------------------

clc; clear all;

load('MasterData_USD_2013_2024.mat', 'M');
cols = ["AUS_wine_USD","LivEx100_USD"];
indexNames = cols;
data = M{:, cellstr(cols)};

R = diff(log(data));   % T x 2 monthly log returns
[T, N] = size(R);
fprintf('Australia-only subsample: %d monthly return observations, %d series (%s)\n', ...
    T, N, strjoin(cellstr(indexNames), ', '));

%% 1) Volatility model selection (same guardrail-fixed selector as Table 4/5/6)
OUT = fit_auto_vol_models(R, indexNames);
for i = 1:N
    fprintf('%-14s -> %s\n', indexNames(i), OUT.choice{i});
    disp(OUT.params{i});
end
write_latex_model_table(OUT, indexNames, 'tab_selected_models_AUSonly.tex');

stdResid = OUT.stdz;
names = indexNames;
save('ResidualSTD_AUSonly.mat', 'stdResid', 'names');

%% 2) DCC / ADCC dependence (Australia wine vs. Liv-ex 100)
Qbar = cov(stdResid, 'partialrows');
Sbar = cov(stdResid .* (stdResid < 0), 'partialrows');

opts = optimoptions(@fmincon, 'Display','off', 'Algorithm','sqp', ...
    'TolFun',1e-10, 'TolCon',1e-10, 'MaxFunctionEvaluations',5e4, 'MaxIterations',5e3);

dcc_neglog = @(x) dccNLL_local(x, stdResid, Qbar);
x0 = [0.05; 0.90]; lb = [0;0]; ub = [1;1];
[theta_dcc, nll_dcc, exit_dcc, ~, ~, ~, H_dcc] = ...
    fmincon(dcc_neglog, x0, [1 1], 0.999, [], [], lb, ub, [], opts);
se_dcc = sqrt(diag(H_dcc \ eye(2)) / T);
t_dcc  = theta_dcc ./ se_dcc;
fprintf('[AUS-only DCC]  a=%.4f (t=%.2f), b=%.4f (t=%.2f), a+b=%.4f\n', ...
    theta_dcc(1), t_dcc(1), theta_dcc(2), t_dcc(2), sum(theta_dcc));

adcc_neglog = @(x) adccNLL_local(x, stdResid, Qbar, Sbar);
x0a = [0.05; 0.90; 0.05]; lba=[0;0;0]; uba=[1;1;1];
[theta_adcc, nll_adcc, exit_adcc, ~, ~, ~, H_adcc] = ...
    fmincon(adcc_neglog, x0a, [1 1 0.5], 0.999, [], [], lba, uba, [], opts);
se_adcc = sqrt(diag(H_adcc \ eye(3)) / T);
t_adcc  = theta_adcc ./ se_adcc;
fprintf('[AUS-only ADCC] a=%.4f (t=%.2f), b=%.4f (t=%.2f), g=%.4f (t=%.2f)\n', ...
    theta_adcc(1), t_adcc(1), theta_adcc(2), t_adcc(2), theta_adcc(3), t_adcc(3));

dccTab = table(["a";"b"], theta_dcc, se_dcc, t_dcc, 'VariableNames', {'Param','Estimate','SE','tStat'});
adccTab = table(["a";"b";"g"], theta_adcc, se_adcc, t_adcc, 'VariableNames', {'Param','Estimate','SE','tStat'});
writetable(dccTab, 'tab_dcc_AUSonly.csv');
writetable(adccTab, 'tab_adcc_AUSonly.csv');

%% 3) Spillover (Diebold-Yilmaz), returns and volatility shocks
Z = stdResid - mean(stdResid, 'omitnan');
H = 10; pMin = 1; pMax = 6; crit = 'BIC';

p_ret = selectVARlag_local(Z, pMin, pMax, crit);
[VARr, SigU_r] = varEstimate_local(Z, p_ret);
ThetaH_r = gfevd_local(VARr, SigU_r, H);
ConnRET = connectedness_local(ThetaH_r, names);
fprintf('[AUS-only RET spillover] VAR order p=%d, TSI=%.2f%%\n', p_ret, ConnRET.TSI);
disp(ConnRET.table);

Z2 = (Z.^2) - mean(Z.^2, 'omitnan');
p_vol = selectVARlag_local(Z2, pMin, pMax, crit);
[VARv, SigU_v] = varEstimate_local(Z2, p_vol);
ThetaH_v = gfevd_local(VARv, SigU_v, H);
ConnVOL = connectedness_local(ThetaH_v, names);
fprintf('[AUS-only VOL spillover] VAR order p=%d, TSI=%.2f%%\n', p_vol, ConnVOL.TSI);
disp(ConnVOL.table);

writetable(ConnRET.table, 'tab_spillover_AUSonly_RET.csv');
writetable(ConnVOL.table, 'tab_spillover_AUSonly_VOL.csv');

%% ====================== local helper functions ======================
function nll = dccNLL_local(par, u, Qbar)
    a = par(1); b = par(2); [T,~] = size(u); Q = Qbar; nll = 0;
    for t = 1:T
        if t>1, ut = u(t-1,:)'; Q = (1-a-b)*Qbar + a*(ut*ut') + b*Q; end
        D = diag(1./sqrt(diag(Q))); Rm = D*Q*D; ut_t = u(t,:)';
        nll = nll + 0.5*( log(det(Rm)) + ut_t'*(Rm\ut_t) );
    end
end

function nll = adccNLL_local(par, u, Qbar, Sbar)
    a = par(1); b = par(2); g = par(3); [T,~] = size(u); Q = Qbar; nll = 0;
    for t = 1:T
        if t>1
            ut = u(t-1,:)'; u_neg = ut .* (ut<0);
            Q = (1-a-b-g)*Qbar + a*(ut*ut') + b*Q + g*(u_neg*u_neg');
        end
        D = diag(1./sqrt(diag(Q))); Rm = D*Q*D; ut_t = u(t,:)';
        nll = nll + 0.5*( log(det(Rm)) + ut_t'*(Rm\ut_t) );
    end
end

function p = selectVARlag_local(Y, pMin, pMax, crit)
    [T,~] = size(Y); best = inf; p = pMin;
    for k = pMin:pMax
        [~, SigU] = varEstimate_local(Y, k);
        logL = - (T-k)*0.5*( size(Y,2)*log(2*pi) + log(det(SigU)) + size(Y,2) );
        kpars = size(Y,2)^2 * k;
        AIC = -2*logL + 2*kpars; BIC = -2*logL + kpars*log(T-k);
        val = strcmpi(crit,'AIC')*AIC + strcmpi(crit,'BIC')*BIC;
        if val < best, best=val; p=k; end
    end
end

function [VAR, SigU] = varEstimate_local(Y, p)
    [T,N] = size(Y); X = [];
    for j=1:p, X = [X, Y((p-j+1):(T-j), :)]; end %#ok<AGROW>
    Yt = Y((p+1):T, :); X = [ones(T-p,1), X];
    B = (X' * X) \ (X' * Yt); U = Yt - X * B;
    SigU = (U' * U) / (T - p - (1 + N*p));
    VAR.B = B; VAR.N = N; VAR.p = p; VAR.SigU = SigU;
end

function ThetaH = gfevd_local(VAR, SigU, H)
    N = VAR.N; p = VAR.p; B = VAR.B;
    A = zeros(N*p); A(1:N, :) = B(2:end, :)';
    if p>1, A((N+1):end, 1:(N*(p-1))) = eye(N*(p-1)); end
    J = [eye(N), zeros(N, N*(p-1))];
    Psi = cell(H,1); Psi{1} = eye(N);
    for h=2:H, Psi{h} = J * (A^(h-1)) * J'; end
    s = diag(SigU).^0.5; ThetaH = zeros(N);
    for i=1:N
        for j=1:N
            num = 0; den = 0;
            for h=1:H
                e_i = Psi{h}(i,:);
                num = num + (e_i * SigU(:,j))^2 / (s(j)^2);
                den = den + (e_i * SigU * e_i');
            end
            ThetaH(i,j) = num / den;
        end
    end
    ThetaH = ThetaH ./ sum(ThetaH,2);
end

function OUTc = connectedness_local(Theta, names)
    N = size(Theta,1);
    offdiagSum = sum(Theta, 'all') - sum(diag(Theta));
    TSI = 100 * offdiagSum / N;
    TO   = 100 * (sum(Theta,1)' - diag(Theta));
    FROM = 100 * (sum(Theta,2) - diag(Theta));
    NET  = TO - FROM;
    names = string(names(:));
    Ttbl = table(names, FROM, TO, NET, 'VariableNames', {'Series','FROM','TO','NET'});
    OUTc = struct('Theta',Theta, 'TSI',TSI, 'TO',TO, 'FROM',FROM, 'NET',NET, 'table',Ttbl);
end
