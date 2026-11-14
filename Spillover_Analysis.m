%% =======================================================================
%  From Dependence to Spillovers: DY Connectedness (returns & volatility)
%  - VAR(p) chosen by AIC/BIC
%  - Generalized FEVD (Pesaran–Shin) at horizon H
%  - Diebold–Yilmaz TO/FROM/NET, Total Spillover Index (TSI)
%  - Optional rolling window
%  -----------------------------------------------------------------------
%  Inputs (in workspace):
%     Z      : T x N matrix of standardized shocks (FIGARCH/GARCH-filtered)
%     names  : 1 x N string/cellstr (optional)
%  Author: (you)
% ========================================================================

clearvars -except Z names; clc;
load('ResidualSTD.mat')   % provides stdResid and names (was hardcoded here, out of sync with rebuild_residualstd.m's actual series list -- now taken from the file itself)
Z=stdResid;
%% -------- settings --------
H      = 10;          % FEVD forecast horizon
pMin   = 1; pMax = 6; % VAR lag selection range
crit   = 'BIC';       % 'AIC' or 'BIC' for lag selection
doRoll = true;        % rolling connectedness
W      = 36;          % rolling window length (months)
Hroll  = 10;          % horizon for rolling FEVD (can match H)

%% -------- inputs & basic checks --------
assert(exist('Z','var')==1 && ~isempty(Z), 'Z (T x N) standardized shocks required.');
[T,N] = size(Z);
if ~(exist('names','var')==1) || isempty(names)
    names = "S"+string(1:N);
end
if iscell(names), names = string(names); end
names = names(:)'; %#ok<NASGU>

% De-mean (robust) to avoid small numeric drift
Z = Z - mean(Z,'omitnan');

%% ====== 1) VAR order selection on returns shocks (Z) ======
p_ret = selectVARlag(Z, pMin, pMax, crit);
fprintf('[RET] Selected VAR order p = %d by %s\n', p_ret, upper(crit));

%% ====== 2) Estimate VAR(p) and generalized FEVD ======
[VARr, SigU_r] = varEstimate(Z, p_ret);                 % coefficients & residual cov
ThetaH_r       = gfevd(VARr, SigU_r, H);                % N x N FEVD at horizon H
ConnRET        = connectedness(ThetaH_r, names);        % DY table (TO/FROM/NET/TSI)
disp('=== Connectedness (RETURNS shocks) ==='); disp(ConnRET.table);
fprintf('[RET] Total Spillover Index = %.3f%%\n', ConnRET.TSI);
writetable(ConnRET.table, 'tab_spillover_RET_18.csv');

%% ====== 3) Volatility spillovers (use squared shocks) ======
Z2 = (Z.^2) - mean(Z.^2,'omitnan');                     % center for stationarity
p_vol = selectVARlag(Z2, pMin, pMax, crit);
fprintf('[VOL] Selected VAR order p = %d by %s\n', p_vol, upper(crit));

[VARv, SigU_v] = varEstimate(Z2, p_vol);
ThetaH_v       = gfevd(VARv, SigU_v, H);
ConnVOL        = connectedness(ThetaH_v, names);
disp('=== Connectedness (VOLTILITY shocks, squared z) ==='); disp(ConnVOL.table);
fprintf('[VOL] Total Spillover Index = %.3f%%\n', ConnVOL.TSI);
writetable(ConnVOL.table, 'tab_spillover_VOL_18.csv');

%% ====== 4) Optional: rolling connectedness over time ======
if doRoll
    [TSI_ret, dates_ret] = rollingTSI(Z,  W, Hroll, pMin, pMax, crit); %#ok<ASGLU>
    [TSI_vol, dates_vol] = rollingTSI(Z2, W, Hroll, pMin, pMax, crit); %#ok<ASGLU>
    % BUG NOTE (2026-09-03, not fixed here): rollingTSI's second output is
    % a plain sample-index vector (1:T)', not a datenum, despite the
    % variable name -- datetick('x','yyyy') on it would be meaningless.
    % Plotting against sample index instead until a real date vector is
    % threaded through; this is diagnostic-only and already flagged
    % elsewhere as underidentified/near-singular at 18-series width.
    fRoll = figure('Name','Rolling Total Spillover Index');
    plot(TSI_ret, '-', 'LineWidth',1.4); hold on;
    plot(TSI_vol, '--', 'LineWidth',1.4);
    grid on; xlabel('Month index (rolling window, not a calendar date)'); ylabel('Total Spillover Index (%)');
    legend({'Returns','Volatility'}, 'Location','best');
    exportgraphics(fRoll, 'fig_spillover_rollingTSI.png', 'Resolution', 200);
    fprintf('Saved fig_spillover_rollingTSI.png\n');
end

%% ====== 5) Pretty panels (optional) ======
% Adjacency-style heatmaps of pairwise contributions (RET and VOL)
% BUG FIX (2026-09-03): xticklabels(names)/yticklabels(names) alone
% assigns `names` positionally to whatever default tick positions imagesc
% happens to pick for an 18x18 matrix (MATLAB's auto tick spacing, which
% is NOT guaranteed to be one tick per row/column once N is this large) --
% silently mislabeling axes rather than just truncating them. Must set
% ticks explicitly to 1:N first so each label lands on its own row/column.
fRET = figure('Name','FEVD Contributions (RET)','Position',[100 100 800 700]);
imagesc(100*ThetaH_r ./ sum(ThetaH_r,2)); colorbar;
set(gca,'XTick',1:N,'XTickLabel',names,'YTick',1:N,'YTickLabel',names,'FontSize',7);
xtickangle(45);
exportgraphics(fRET, 'fig_spillover_fevd_RET.png', 'Resolution', 200);
fprintf('Saved fig_spillover_fevd_RET.png\n');

fVOL = figure('Name','FEVD Contributions (VOL)','Position',[100 100 800 700]);
imagesc(100*ThetaH_v ./ sum(ThetaH_v,2)); colorbar;
set(gca,'XTick',1:N,'XTickLabel',names,'YTick',1:N,'YTickLabel',names,'FontSize',7);
xtickangle(45);
exportgraphics(fVOL, 'fig_spillover_fevd_VOL.png', 'Resolution', 200);
fprintf('Saved fig_spillover_fevd_VOL.png\n');

%% ====================== FUNCTIONS ======================

function p = selectVARlag(Y, pMin, pMax, crit)
% Select VAR lag p in [pMin,pMax] by AIC or BIC.
    [T,~] = size(Y);
    best = inf; p = pMin;
    for k = pMin:pMax
        [~, SigU] = varEstimate(Y, k);
        % log-likelihood under Gaussian VAR
        logL = - (T-k)*0.5*( size(Y,2)*log(2*pi) + log(det(SigU)) + size(Y,2) );
        kpars = size(Y,2)^2 * k;  % rough dof: each lag has N^2 parameters
        AIC = -2*logL + 2*kpars;
        BIC = -2*logL + kpars*log(T-k);
        val = strcmpi(crit,'AIC')*AIC + strcmpi(crit,'BIC')*BIC;
        if val < best, best=val; p=k; end
    end
end

function [VAR, SigU] = varEstimate(Y, p)
% OLS estimate of VAR(p): Y_t = c + A1 Y_{t-1} + ... + Ap Y_{t-p} + u_t
    [T,N] = size(Y);
    X = []; 
    for j=1:p
        X = [X, Y((p-j+1):(T-j), :)]; %#ok<AGROW>
    end
    Yt = Y((p+1):T, :);
    X  = [ones(T-p,1), X];                 % intercept
    % OLS stacked by equation
    B = (X' * X) \ (X' * Yt);              % (1+Np) x N
    U = Yt - X * B;                        % residuals
    SigU = (U' * U) / (T - p - (1 + N*p)); % covariance of residuals
    VAR.B = B; VAR.N = N; VAR.p = p; VAR.SigU = SigU;
end

function ThetaH = gfevd(VAR, SigU, H)
% Generalized FEVD (Pesaran-Shin, 1998) at horizon H
% Returns ThetaH: N x N matrix; row i: fraction of i's H-step forecast error
% variance due to shocks in j.
    N = VAR.N; p = VAR.p; B = VAR.B;
    % Companion form to get MA representation
    A = zeros(N*p);
    A(1:N, :) = B(2:end, :)';             % strip intercept; pack lags
    if p>1
        A((N+1):end, 1:(N*(p-1))) = eye(N*(p-1));
    end
    J = [eye(N), zeros(N, N*(p-1))];
    % MA coefficients Psi(h): N x N for h=0...(H-1)
    Psi = cell(H,1);
    Psi{1} = eye(N);
    for h=2:H
        Psi{h} = J * (A^(h-1)) * J';
    end
    % Generalized FEVD per Pesaran–Shin
    s = diag(SigU).^0.5;                   % std of each shock
    ThetaH = zeros(N);
    for i=1:N
        for j=1:N
            num = 0;
            den = 0;
            for h=1:H
                e_i = Psi{h}(i,:);         % 1 x N
                num = num + (e_i * SigU(:,j))^2 / (s(j)^2);
                den = den + (e_i * SigU * e_i');
            end
            ThetaH(i,j) = num / den;
        end
    end
    % Normalize rows to sum to 1
    ThetaH = ThetaH ./ sum(ThetaH,2);
end

function OUT = connectedness(Theta, names)
% Diebold–Yilmaz connectedness from FEVD Theta (N x N, rows sum to 1)
    N = size(Theta,1);
    offdiagSum = sum(Theta, 'all') - sum(diag(Theta));
    TSI = 100 * offdiagSum / N;                 % Total Spillover Index (%)
    TO   = 100 * (sum(Theta,1)' - diag(Theta)); % contributions sent to others
    FROM = 100 * (sum(Theta,2) - diag(Theta));  % received from others
    NET  = TO - FROM;                            % net transmitter (>0) or receiver

    % table
    names = string(names(:));
    Ttbl = table(names, FROM, TO, NET, 'VariableNames', {'Series','FROM','TO','NET'});
    OUT = struct('Theta',Theta, 'TSI',TSI, 'TO',TO, 'FROM',FROM, 'NET',NET, 'table',Ttbl);
end

function [TSI, tvec] = rollingTSI(Y, W, H, pMin, pMax, crit)
% Rolling Total Spillover Index (returns a vector aligned to the end of window)
    T = size(Y,1);
    TSI = NaN(T,1);
    tvec = (1:T)'; %#ok<NASGU>
    for t = W:T
        Yw = Y((t-W+1):t, :);
        p = selectVARlag(Yw, pMin, pMax, crit);
        [VARw, SigUw] = varEstimate(Yw, p);
        Theta = gfevd(VARw, SigUw, H);
        C = connectedness(Theta, 1:size(Y,2));
        TSI(t) = C.TSI;
    end
    % Build a date vector if your Z had timestamps; here we fallback to serial months
    tvec = (1:T)';
end
