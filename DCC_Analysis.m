%% Country-Specific Wine Index Analysis
%%

% This script performs an international comparative study of hybrid
% hedonic–repeat‐sales wine indices for South Africa, Argentina, and
% Australia against the Liv-ex 100 benchmark. It computes key performance
% metrics and uses a manual DCC-GARCH(1,1) model to assess volatility
% clustering and dynamic correlations.
%
% Reference:
% Fogarty, J. J., & Jones, C. (2011). Return to wine: A comparison of the
% hedonic, repeat sales and hybrid approaches. Australian Economic Papers,
% 50(4), 147–156.
%
% Data requirements:
%   - 'portfolio.mat' containing:
%       • Data: [T×N] matrix of  N indices
%       • Dates: [T×1] datetime vector
%   - Index order: 1:SAfw10, 2:Argfw, 3:Ausfw, 4:chifw, 5:STX40, 6:ETF500,
%                  7:STXEMG, 8:STXWDM, 9:ETFGLD, 10:ETFBND,
%                  11:STXPRO, 12:Liv-ex100
%   - Full Name:  1: South Africa Fine wine, 2: Argentina Fine Wine, 
%                 3: Australia Fine Wine, 4: Chile Fine Wine, 
%                 5: Satrix Top 40 ETF, 6: 1nvest S&P 500 Index ETF, 
%                 7: Satrix MSCI Emerging Markets ETF, 
%                 8:Satrix MSCI World ETF 9: 1nvest Gold ETF,
%                10: 1nvest SA Bond ETF, 11: Satrix Property Portfolio ETF,
%                12: Liv Ex Fine Wine 100
%
% Output:
%   - Table of annualized return, volatility, and Sharpe ratio
%   - Static correlation matrix
%   - Plot of time-varying correlations with Liv-ex100
%   - Saved MAT and CSV results
% -------------------------------------------------------------------------

% Clean workspace
clearvars; close all; clc

%% Load Data
load('projectdata.mat');
load('projectdate.mat'); % T×1 datetime

% Data=table2array(data);
R   = diff(log(data)); % T×N returns
[T, N] = size(R);

%% Performance Metrics
% Frequency and risk-free rate
frequency   = 12;    % e.g., 12 for monthly data
riskFreeAnn = 0.01;  % annual risk-free rate

% Annualized returns (arithmetic approximation)
annualReturn = mean(R) * frequency;

% Annualized volatility
annualVolatility = std(R) * sqrt(frequency);

% Sharpe ratio (excess return / volatility)
sharpeRatio = (annualReturn - riskFreeAnn) ./ annualVolatility;

% Static correlation matrix of returns
correlationMatrix = corrcoef(R);

% Prepare and display results table
indexNames = ["SAfw","Ausfw","Argfw","chifw","SP500", ...
              "MSCIEM","Bond","Gold","Liv-ex100"]';
PerfTable = table( indexNames, annualReturn', annualVolatility', sharpeRatio', ...
    'VariableNames', {'Index', 'AnnReturn', 'AnnVolatility', 'SharpeRatio'} );
disp('=== Performance Metrics ===');
disp(PerfTable);
disp('=== Static Correlation Matrix ===');
disp(correlationMatrix);


%% Univariate GARCH(1,1) Fitting
% Preallocate
stdResid   = nan(T, N);   % standardized residuals
condVarMod = nan(T, N);   % conditional variances
garchModel = garch(1,1);

for i = 1:N
    % Smooth data to remove occasional spikes
%      series = smoothdata(R(:,i), 'gaussian', 3);
     series = R(:,i);
    % Estimate GARCH(1,1)
    estMdl = estimate(garchModel, series, 'Display', 'off');
    
    % Infer residuals and conditional variance
    [resid, varH] = infer(estMdl, series);
    stdResid(:,i)   = resid ./ sqrt(varH);
    condVarMod(:,i) = varH;
end
%% Univariate GJR-GARCH(1,1) Fitting
% R: T x N matrix of (mean-adjusted) returns
[T, N] = size(R);

stdResid   = nan(T, N);   % standardized residuals
condVarMod = nan(T, N);   % conditional variances

% Base variance model: h_t = k + alpha*e_{t-1}^2 + gamma*e_{t-1}^2*I(e_{t-1}<0) + beta*h_{t-1}
gjrModel = gjr(1,1);      % Econometrics Toolbox

% (Optional) optimizer options for tougher series
try
    opt = optimoptions('fmincon','Display','off','MaxFunctionEvaluations',1e5,'MaxIterations',5e3);
catch
    opt = []; % older MATLAB: silently ignore
end

for i = 1:N
    series = R(:,i);
    % Drop leading/trailing NaNs (keep an index to reinsert in place)
    idxGood = ~isnan(series);
    y = series(idxGood);

    if numel(y) < 30
        warning('Series %d has too few observations for GJR-GARCH. Skipping.', i);
        continue;
    end

    % Estimate GJR-GARCH(1,1). Student-t innovations are common for financial returns.
    try
        estMdl = estimate(gjrModel, y, ...
            'Display','off', ...
            'Options',opt);
    catch ME
        % Fallback: try Gaussian innovations
        warning('GJR-GARCH(1,1) (t) failed for series %d (%s). Retrying with Gaussian.', i, ME.message);
        estMdl = estimate(gjrModel, y, ...
            'Display','off','Options',opt);
    end

    % Infer residuals and conditional variance
    [resid, varH] = infer(estMdl, y);

    % Store aligned to original length
    z = nan(T,1); vh = nan(T,1);
    z(idxGood)  = resid ./ sqrt(varH);
    vh(idxGood) = varH;

    stdResid(:,i)   = z;
    condVarMod(:,i) = vh;
end
%% FIGARCH(1,d,1) fitting by Gaussian QML
% R: T x N matrix of (mean-adjusted) returns
%  m=mean(R);
% Ret=R-m;
% R=Ret;
[T, N] = size(R);
stdResid   = nan(T, N);
condVarMod = nan(T, N);

% Options
Kmax   = min(1500, max(300, T-1));  % truncation length for fractional weights
burn   = 50;                        % burn-in when summing loglik
opts   = optimoptions('fmincon','Display','off','MaxIterations',2000,'MaxFunctionEvaluations',2e5);
have_fmincon = ~isempty(which('fmincon'));

for i = 1:N
    y = R(:,i);
    good = isfinite(y);
    y = y(good);
%     if numel(y) < 200
%         warning('Series %d: too short for reliable FIGARCH; need ~200+ points.', i);
%         continue;
%     end

    % Initial guesses (omega, phi, beta, d)
    s2   = var(y,'omitnan');
    p0   = [0.05*s2, 0.2, 0.6, 0.3];  % rough start
    lb   = [1e-12,   0.0, 0.0, 1e-6];
    ub   = [10*s2,   0.999, 0.999, 0.999];

    % Wrap objective
    obj = @(p) figarch_nll_gauss(y, p, Kmax, burn);

    if have_fmincon
        [pHat, fval,EXITFLAG,OUTPUT,LAMBDA,GRAD,HESSIAN] = fmincon(obj, p0, [],[],[],[], lb, ub, [], opts);
         Stderr=(1.0/sqrt(65)).*sqrt(diag(inv(HESSIAN)));

    else
        % Fallback: fminsearch with soft bounds penalty
        penObj = @(p) obj(softbox(p,lb,ub));
        pFree0 = invsoftbox(p0,lb,ub);
        pFree  = fminsearch(penObj, pFree0, optimset('Display','off','MaxFunEvals',2e5,'MaxIter',2e4));
        pHat   = softbox(pFree,lb,ub);
    end

    % Infer conditional variance and standardized residuals at pHat
    [~, sigma2, eps] = figarch_nll_gauss(y, pHat, Kmax, burn);
    z = eps ./ sqrt(sigma2);

    % Reinsert into T x 1 holders
    zfull  = nan(T,1); vfull = nan(T,1);
    zfull(good)  = z;
    vfull(good)  = sigma2;

    stdResid(:,i)   = zfull;
    condVarMod(:,i) = vfull;

    fprintf('Series %d FIGARCH(1,d,1): omega=%.4g, phi=%.3f, beta=%.3f, d=%.3f\n', ...
            i, pHat(1), pHat(2), pHat(3), pHat(4));
    Params(i,:)=[pHat(1) pHat(2) pHat(3) pHat(4)];
    pval(i,:)=[pHat(1)/Stderr(1) pHat(2)/Stderr(2) pHat(3)/Stderr(3) pHat(4)/Stderr(4)];
end

%% 3. Precompute unconditional matrices
Qbar = cov(stdResid, 'partialrows');               
Sbar = cov(stdResid .* (stdResid < 0), 'partialrows'); % for asymmetry

%% 4. Estimate DCC parameters [a, b] by QML
% Initial guess and bounds
x0_dcc = [0.5; 0.5];
lb_dcc = [0; 0];          % a >= 0, b >= 0
ub_dcc = [1; 1];          % a <=1, b <=1 
A_dcc  = [1, 1]; b_dcc = 0.99999;  % a + b <= 0.999

% opts = optimoptions('fmincon','Algorithm','sqp','Display','iter');
opts = optimoptions(@fmincon,'Display','iter','TolFun',1e-10,'TolCon',1e-10,'MaxFunctionEvaluations',35000,'Algorithm','sqp');
dcc_neglog = @(x) dccNegLogLikelihood(x, stdResid, Qbar);

[theta_dcc, fval_dcc,EXITFLAG,OUTPUT,LAMBDA,GRAD,HESSIAN] = fmincon( dcc_neglog, x0_dcc, A_dcc, b_dcc, [],[], lb_dcc, ub_dcc, [], opts );
a_dcc = theta_dcc(1);  b_dcc = theta_dcc(2);
Stderr=(1.0/sqrt(65)).*sqrt(diag(inv(HESSIAN)));
ta_dcc=theta_dcc(1)/Stderr(1);tb_dcc=theta_dcc(2)/Stderr(2);
fprintf('\nDCC Estimates: a = %.4f, b = %.4f\n', a_dcc, b_dcc);

%% 5. Estimate ADCC parameters [a, b, g] by QML
x0_adcc = [0.5;0.3;0.2];
lb_adcc = [0;0;0];
ub_adcc = [1;1;1];
A_adcc  = [1, 1, 1]; b_adcc = 0.99;

adcc_neglog = @(x) adccNegLogLikelihood(x, stdResid, Qbar, Sbar);

[theta_adcc, fval_adcc] = fmincon( adcc_neglog, x0_adcc, A_adcc, b_adcc, [],[], lb_adcc, ub_adcc, [], opts );
a_adcc = theta_adcc(1);  b_adcc = theta_adcc(2);  g_adcc = theta_adcc(3);

fprintf('ADCC Estimates: a = %.4f, b = %.4f, g = %.4f\n', a_adcc, b_adcc, g_adcc);


%% 6. Compute dynamic correlations using estimated parameters
Rt_dcc  = computeDCC(stdResid, Qbar, a_dcc, b_dcc);
Rt_adcc = computeADCC(stdResid, Qbar, Sbar, a_adcc, b_adcc, g_adcc);

%% 7. Plot correlation vs benchmark (last column)
bench = N;
others = 1:N-1;
 
figure('Color','w');
subplot(2,1,1); hold on;
for j = others
    plot(date(2:end), squeeze(Rt_dcc(j,bench,:)), 'LineWidth',1.2);
end
hold off; grid on; datetick('x','yyyy');
xlabel('Year'); ylabel('Corr (DCC)'); title('DCC-FIGARCH(1,1)');
legend(indexNames(others),'Location','eastoutside');

subplot(2,1,2); hold on;
for j = others
    plot(date(2:end), squeeze(Rt_adcc(j,bench,:)), 'LineWidth',1.2);
end
hold off; grid on; datetick('x','yyyy');
xlabel('Year'); ylabel('Corr (ADCC)'); title('ADCC-FIGARCH(1,1)');
legend(indexNames(others),'Location','eastoutside');
%% Manual DCC-FIGARCH(1,1) Recursion
% DCC parameters (must satisfy a + b < 1)
a = theta_dcc(1);
b = theta_dcc(2);
assert(a + b < 1, 'DCC parameters must satisfy a + b < 1');

% Unconditional covariance of standardized residuals
Qbar = cov(stdResid, 'partialrows');
Q     = Qbar;
Rt    = NaN(N, N, T);   % time-varying correlation matrices

for t = 1:T
    if t > 1
        u_prev = stdResid(t-1, :)';
        Q = (1 - a - b) * Qbar + a * (u_prev * u_prev') + b * Q;
    end
    % Normalize to correlation matrix
    D = diag(1 ./ sqrt(diag(Q)));
    Rt(:,:,t) = D * Q * D;
end

%% Plot Dynamic Correlations with Liv-ex100
benchmarkIdx = N;             % Liv-ex100 is the last column
 countryIdx   = 1:(N-1);
% countryIdx   = 1:4;

figure('Color','w');
hold on;
for j = countryIdx
    plot(date(2:end), squeeze(Rt(j, benchmarkIdx, :)), 'LineWidth', 1.2);
end
hold off;
grid on;
datetick('x','yyyy');
xlabel('Year');
ylabel('Correlation with Liv-ex100');
axis tight
legend(indexNames(countryIdx), 'Location', 'EastOutside');
title('DCC-FIGARCH(1,1): Dynamic Correlations with Liv-ex100');

%% Save Results
save('Performance_and_DCC.mat', 'PerfTable', 'correlationMatrix', ...
     'condVarMod', 'stdResid', 'Rt', 'date', 'indexNames');
writetable(PerfTable, 'PerformanceMetrics.csv');

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


% Figarch

function [nll, sigma2, eps] = figarch_nll_gauss(y, p, K, burn)
% Gaussian QML for FIGARCH(1,d,1): params p = [omega, phi, beta, d]
% Recursion: (1 - beta L) h_t = omega + [1 - beta L - (1 - phi L)(1 - L)^d] * e_{t-1}^2
% Truncate fractional weights at K lags.

    omega = p(1); phi = p(2); beta = p(3); d = p(4);
    T = numel(y);
    eps = y;                      % innovations (mean-adjusted input)
    % Precompute fractional weights for (1-L)^d
    w = fracdiff_weights(d, K);   % w(1)=1, w(k+1) ~ coeff on L^k
    % Build lambda_k for k>=1:   lambda(L) = 1 - beta L - (1 - phi L)(1-L)^d
    % Expand (1 - phi L)(1-L)^d = w_0 + sum_{k>=1} (w_k - phi * w_{k-1}) L^k
    % Hence, for k>=1: lambda_k = -beta*1_{k=1} - (w_k - phi*w_{k-1})
    lambda = zeros(K,1);  % lambda(1) corresponds to L^1 coefficient
    % k = 1:
    lambda(1) = -beta - (w(2) - phi*w(1));
    % k >= 2:
    for k = 2:K
        lambda(k) = -( w(k+1) - phi*w(k) );
    end
    % Note: constant term works out via recursion as written below.

    % Initialize variance with unconditional-ish level
    h = nan(T,1);
    h0 = var(y,'omitnan');
    h(1:max(2,burn)) = max(1e-10, h0);

    % Compute h_t forward using truncated infinite sum
    % (1 - beta L) h_t = omega + sum_{k=1}^K lambda_k * e_{t-k}^2
    % => h_t = beta * h_{t-1} + omega + sum_{k=1}^{min(K,t-1)} lambda_k * e_{t-k}^2
    for t = max(2,burn+1):T
        acc = 0.0;
        m = min(K, t-1);
        % Accumulate lambda_k * e_{t-k}^2
        for k = 1:m
            ek = eps(t-k);
            if ~isfinite(ek), ek = 0; end
            acc = acc + lambda(k) * (ek*ek);
        end
        ht = beta * h(t-1) + omega + acc;
        h(t) = max(ht, 1e-12); % positivity guard
    end

    sigma2 = h;
    % Gaussian QML log-likelihood
    valid = isfinite(sigma2) & sigma2 > 0;
    % Start from burn+1 to reduce initialization bias
    s = max(burn+1, 2);
    L = 0.5*( log(sigma2(s:end)) + (eps(s:end).^2) ./ sigma2(s:end) );
    nll = sum(L(valid(s:end)));
end

function w = fracdiff_weights(d, K)
% Coefficients for (1 - L)^d = sum_{k=0}^\infty w_k L^k, truncated at K
% Recurrence: w_0 = 1; w_k = w_{k-1} * (k-1 - d)/k
    w = zeros(K+1,1);
    w(1) = 1.0;  % w_0
    for k = 2:K+1
        w(k) = w(k-1) * ((k-2) - d) / (k-1);
    end
end

