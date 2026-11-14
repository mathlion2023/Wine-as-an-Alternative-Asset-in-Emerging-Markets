% model
clc; clear all;
% Repointed (2026-08-25) at the USD master dataset (Build_MasterData_USD.m
% output), Jan 2013-Nov 2024, replacing projectdata.mat/projectdate.mat
% (Jun 2019-Nov 2024, 66 months). Runs on ALL 18 series in the master
% dataset (2026-08-25, per user instruction), not just the original
% manuscript's 9-asset subset.
load('MasterData_USD_2013_2024.mat', 'M');
srcCols    = M.Properties.VariableNames(2:end);
indexNames = string(srcCols)';
data = M{:, srcCols};
date = datenum(M.Period); % T×1 datenum

% Data=table2array(data);
R   = diff(log(data)); % T×N returns
[T, N] = size(R);
% R: T x N matrix of returns (monthly or daily). Example names:
%% Fit & select per series
OUT = fitIndices_autoModels(R, indexNames);

%% Inspect choices
for i = 1:numel(indexNames)
    fprintf('%-10s -> %s | logL=%.1f, AIC=%.1f, BIC=%.1f\n', ...
        indexNames(i), OUT.choice{i}, OUT.stats{i}.logL, OUT.stats{i}.AIC, OUT.stats{i}.BIC);
end

%% Example: standardized residual checks for series 1
i = 1;
figure; tiledlayout(2,1);
nexttile; plot(OUT.stdResid(:,i)); title(indexNames(i) + " standardized residuals"); grid on;
nexttile; plot(sqrt(OUT.condVar(:,i))); title(indexNames(i) + " conditional stdev"); grid on;

%% Precompute unconditional matrices
stdResid=OUT.stdResid;   % was `Residual` (undefined) -- OUT.stdResid is fitIndices_autoModels' actual output
Qbar = cov(stdResid, 'partialrows');
Sbar = cov(stdResid .* (stdResid < 0), 'partialrows'); % for asymmetry

%% 4. Estimate DCC parameters [a, b] by QML
% Initial guess and bounds
x0_dcc = [0.45; 0.45];
lb_dcc = [0; 0];          % a >= 0, b >= 0
ub_dcc = [1; 1];          % a <=1, b <=1 
A_dcc  = [1, 1]; b_dcc = 0.999;  % a + b <= 0.999

% opts = optimoptions('fmincon','Algorithm','sqp','Display','iter');
opts = optimoptions(@fmincon,'Display','iter','TolFun',1e-10,'TolCon',1e-10,'MaxFunctionEvaluations',35000,'Algorithm','sqp');
dcc_neglog = @(x) dccNegLogLikelihood(x, stdResid, Qbar);

[theta_dcc, fval_dcc,EXITFLAG,OUTPUT,LAMBDA,GRAD,HESSIAN] = fmincon( dcc_neglog, x0_dcc, A_dcc, b_dcc, [],[], lb_dcc, ub_dcc, [], opts );
a_dcc = theta_dcc(1);  b_dcc = theta_dcc(2);
Stderr=(1.0/sqrt(T)).*sqrt(diag(inv(HESSIAN)));   % was hardcoded sqrt(65) for the old 66-month sample; T is now 142
ta_dcc=theta_dcc(1)/Stderr(1);tb_dcc=theta_dcc(2)/Stderr(2);
fprintf('\nDCC Estimates: a = %.4f, b = %.4f\n', a_dcc, b_dcc);

%% 5. Estimate ADCC parameters [a, b, g] by QML
x0_adcc = [0.5;0.13;0.2];
lb_adcc = [0;0;0];
ub_adcc = [1;1;1];
A_adcc  = [1, 1, 1]; b_adcc = 0.999999;

adcc_neglog = @(x) adccNegLogLikelihood(x, stdResid, Qbar, Sbar);

[theta_adcc, fval_adcc] = fmincon( adcc_neglog, x0_adcc, A_adcc, b_adcc, [],[], lb_adcc, ub_adcc, [], opts );
a_adcc = theta_adcc(1);  b_adcc = theta_adcc(2);  g_adcc = theta_adcc(3);

fprintf('ADCC Estimates: a = %.4f, b = %.4f, g = %.4f\n', a_adcc, b_adcc, g_adcc);


%% 6. Compute dynamic correlations using estimated parameters
Rt_dcc  = computeDCC(stdResid, Qbar, a_dcc, b_dcc);
Rt_adcc = computeADCC(stdResid, Qbar, Sbar, a_adcc, b_adcc, g_adcc);

%% 7. Plot correlation vs benchmark (Liv-ex100)
% Was `bench = N` assuming Liv-ex100 was the last column -- no longer true
% now that all 18 series are loaded (Gold_USD is last); found by name instead.
bench  = find(indexNames == "LivEx100_USD");
others = setdiff(1:N, bench);
 
figure('Color','w');
 hold on;
for j = others
    plot(date(2:end), squeeze(Rt_dcc(j,bench,:)), 'LineWidth',1.2);
end
hold off; grid on; datetick('x','yyyy');
xlabel('Year'); ylabel('Corr (DCC)'); %title('DCC-FIGARCH(1,1)');
legend(indexNames(others),'Location','eastoutside');
axis tight;
%%
 hold on;
for j = others
    plot(date(2:end), squeeze(Rt_adcc(j,bench,:)), 'LineWidth',1.2);
end
hold off; grid on; datetick('x','yyyy');
xlabel('Year'); ylabel('Corr (ADCC)'); %title('ADCC-FIGARCH(1,1)');
legend(indexNames(others),'Location','eastoutside');
axis tight;
%% Plot Dynamic Correlations with Liv-ex100
benchmarkIdx = bench;          % Liv-ex100, found by name above
countryIdx   = others;

figure('Color','w');
hold on;
for j = countryIdx
    plot(date(2:end), squeeze(Rt_dcc(j, benchmarkIdx, :)), 'LineWidth', 1.2);   % was undefined `Rt` -- this script never computed one, only Rt_dcc/Rt_adcc above
end
hold off;
grid on;
datetick('x','yyyy');
xlabel('Year');
ylabel('Correlation with Liv-ex100');
axis tight
legend(indexNames(countryIdx), 'Location', 'EastOutside');
title('DCC-FIGARCH(1,1): Dynamic Correlations with Liv-ex100');
axis tight;
%% Save Results
% PerfTable/correlationMatrix/condVarMod were never computed by this
% script (that block was copy-pasted from DCC_Analysis.m) -- saving what
% this script actually produces instead.
save('Performance_and_DCC.mat', 'OUT', 'stdResid', 'Rt_dcc', 'Rt_adcc', 'date', 'indexNames');

%% Acknowledgements
% Data for Australia, Argentina, and Chile were kindly provided by Dr. Gertjan Verdickt (University of Auckland).
% Analysis implemented in MATLAB by Prof. Mesias Alfeus, Stellenbosch University.


%% 8. Nested Functions for QML objectives and recursions

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





