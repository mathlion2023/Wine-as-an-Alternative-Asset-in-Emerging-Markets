%% CVaR_Sensitivity_4country.m (2026-10-10, peer-review item 8)
% -------------------------------------------------------------------------
% Peer review: "A 95% empirical CVaR estimate based on a 36-month window
% relies on roughly two tail observations. Report sensitivity to
% estimation windows, tail levels, and stress scenarios. Do not present
% the resulting allocation as precisely estimated."
%
% Table 10's static CVaR allocation (Portfolio_Optimization_4country.m)
% uses the FULL SHORT/baseline sample (Tn=65 months) at alpha=0.95, i.e.
% (1-0.95)*65 ~ 3.3 effective tail scenarios -- thin, though less extreme
% than the rolling BACKTEST's 36-month-window CVaR construction the
% review's "roughly two" figure describes (Table 11/12, (1-0.95)*36=1.8).
% This script re-solves the SAME static CVaR program at alpha in
% {0.90, 0.95, 0.975} and at window in {full sample, trailing 36 months
% only}, reporting the resulting wine allocation and CVaR value at each
% setting so the sensitivity is shown, not asserted.
% -------------------------------------------------------------------------

clear; clc;
load('MasterData_USD_2019_2024_4country.mat', 'M');
names = M.Properties.VariableNames(2:end);
N = numel(names);
lvl = M{:, names};
R_full = diff(log(lvl));
[Tn_full, ~] = size(R_full);
wineIdx = find(ismember(names, {'SA_wine_USD','AUS_wine_USD','ARG_wine_USD','CHL_wine_USD'}));

lb = zeros(N,1); ub = ones(N,1);
lpopts = optimoptions('linprog','Display','off');

windows = {'Full sample (65mo)', Tn_full; 'Trailing 36mo', 36};
alphas = [0.90, 0.95, 0.975];

results = table();
for wi = 1:size(windows,1)
    wname = windows{wi,1}; wlen = windows{wi,2};
    R = R_full(end-wlen+1:end, :);
    Tn = size(R,1);
    for ai = 1:numel(alphas)
        alpha = alphas(ai);
        nTailEff = (1-alpha)*Tn;
        [w, cvarVal] = solveCVaR_val(R, N, lb, ub, alpha, lpopts);
        wineAlloc = sum(w(wineIdx));
        results = [results; table(string(wname), Tn, alpha, nTailEff, wineAlloc, cvarVal, ...
            'VariableNames', {'Window','Tn','Alpha','EffTailScenarios','WineAllocation','CVaR_value'})]; %#ok<AGROW>
        fprintf('%-20s Tn=%3d alpha=%.3f (eff. tail n=%.1f): wine alloc=%.3f, CVaR=%.4f\n', ...
            wname, Tn, alpha, nTailEff, wineAlloc, cvarVal);
    end
end
disp(results);
writetable(results, 'tab_cvar_sensitivity_4country.csv');
fprintf('Saved tab_cvar_sensitivity_4country.csv\n');

function [w, cvarVal] = solveCVaR_val(R, N, lb, ub, alpha, lpopts)
    Tn = size(R,1);
    f = [zeros(N,1); 1; ones(Tn,1)/((1-alpha)*Tn)];
    A = [-R, -ones(Tn,1), -eye(Tn)];
    b = zeros(Tn,1);
    Aeq = [ones(1,N), 0, zeros(1,Tn)]; beq = 1;
    lbv = [lb; -Inf; zeros(Tn,1)]; ubv = [ub; Inf; Inf(Tn,1)];
    x = linprog(f, A, b, Aeq, beq, lbv, ubv, lpopts);
    w = x(1:N);
    zeta = x(N+1); u = x(N+2:end);
    cvarVal = zeta + sum(u)/((1-alpha)*Tn);
end
