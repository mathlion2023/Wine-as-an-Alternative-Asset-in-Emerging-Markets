%% Robustness_HedonicAUS_Spillover.m
% -------------------------------------------------------------------------
% Referee 2 item 5 robustness run: core static Diebold-Yilmaz connectedness
% (RET and VOL channels) from Spillover_Analysis.m, run on
% ResidualSTD_HedonicAUS.mat (AUS_wine_USD replaced by the Langton's
% chained-hedonic-index series) instead of ResidualSTD.mat. The rolling-
% window diagnostic and FEVD heatmap figures are intentionally skipped
% here (they are diagnostic/manuscript-figure extras, not needed for the
% Table 8/9-equivalent numeric comparison this robustness check is for)
% -- see Spillover_Analysis.m for those. Outputs are suffixed _HedonicAUS
% so nothing here overwrites the manuscript's actual Table 8/9.
% -------------------------------------------------------------------------
clearvars -except Z names; clc;
load('ResidualSTD_HedonicAUS.mat')
Z = stdResid;

H    = 10;          % FEVD forecast horizon
pMin = 1; pMax = 6; % VAR lag selection range
crit = 'BIC';

[T,N] = size(Z);
if ~(exist('names','var')==1) || isempty(names)
    names = "S"+string(1:N);
end
if iscell(names), names = string(names); end
names = names(:)';

Z = Z - mean(Z,'omitnan');

%% ---- Returns channel ----
p_ret = selectVARlag(Z, pMin, pMax, crit);
fprintf('[RET] Selected VAR order p = %d by %s\n', p_ret, upper(crit));
[VARr, SigU_r] = varEstimate(Z, p_ret);
ThetaH_r       = gfevd(VARr, SigU_r, H);
ConnRET        = connectedness(ThetaH_r, names);
disp('=== Connectedness (RETURNS shocks, HedonicAUS) ==='); disp(ConnRET.table);
fprintf('[RET] Total Spillover Index = %.3f%%\n', ConnRET.TSI);
writetable(ConnRET.table, 'tab_spillover_RET_HedonicAUS.csv');

%% ---- Volatility channel (squared shocks) ----
Z2 = (Z.^2) - mean(Z.^2,'omitnan');
p_vol = selectVARlag(Z2, pMin, pMax, crit);
fprintf('[VOL] Selected VAR order p = %d by %s\n', p_vol, upper(crit));
[VARv, SigU_v] = varEstimate(Z2, p_vol);
ThetaH_v       = gfevd(VARv, SigU_v, H);
ConnVOL        = connectedness(ThetaH_v, names);
disp('=== Connectedness (VOLATILITY shocks, HedonicAUS) ==='); disp(ConnVOL.table);
fprintf('[VOL] Total Spillover Index = %.3f%%\n', ConnVOL.TSI);
writetable(ConnVOL.table, 'tab_spillover_VOL_HedonicAUS.csv');

iAUS = find(names == "AUS_wine_USD", 1);
if ~isempty(iAUS)
    fprintf('\nAUS_wine_USD (hedonic-swapped) NET: RET=%.2f%%, VOL=%.2f%%\n', ...
        ConnRET.NET(iAUS), ConnVOL.NET(iAUS));
end

%% ====================== FUNCTIONS ======================

function p = selectVARlag(Y, pMin, pMax, crit)
    [T,~] = size(Y);
    best = inf; p = pMin;
    for k = pMin:pMax
        [~, SigU] = varEstimate(Y, k);
        logL = - (T-k)*0.5*( size(Y,2)*log(2*pi) + log(det(SigU)) + size(Y,2) );
        kpars = size(Y,2)^2 * k;
        AIC = -2*logL + 2*kpars;
        BIC = -2*logL + kpars*log(T-k);
        val = strcmpi(crit,'AIC')*AIC + strcmpi(crit,'BIC')*BIC;
        if val < best, best=val; p=k; end
    end
end

function [VAR, SigU] = varEstimate(Y, p)
    [T,N] = size(Y);
    X = [];
    for j=1:p
        X = [X, Y((p-j+1):(T-j), :)]; %#ok<AGROW>
    end
    Yt = Y((p+1):T, :);
    X  = [ones(T-p,1), X];
    B = (X' * X) \ (X' * Yt);
    U = Yt - X * B;
    SigU = (U' * U) / (T - p - (1 + N*p));
    VAR.B = B; VAR.N = N; VAR.p = p; VAR.SigU = SigU;
end

function ThetaH = gfevd(VAR, SigU, H)
    N = VAR.N; p = VAR.p; B = VAR.B;
    A = zeros(N*p);
    A(1:N, :) = B(2:end, :)';
    if p>1
        A((N+1):end, 1:(N*(p-1))) = eye(N*(p-1));
    end
    J = [eye(N), zeros(N, N*(p-1))];
    Psi = cell(H,1);
    Psi{1} = eye(N);
    for h=2:H
        Psi{h} = J * (A^(h-1)) * J';
    end
    s = diag(SigU).^0.5;
    ThetaH = zeros(N);
    for i=1:N
        for j=1:N
            num = 0;
            den = 0;
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

function OUT = connectedness(Theta, names)
    N = size(Theta,1);
    offdiagSum = sum(Theta, 'all') - sum(diag(Theta));
    TSI = 100 * offdiagSum / N;
    TO   = 100 * (sum(Theta,1)' - diag(Theta));
    FROM = 100 * (sum(Theta,2) - diag(Theta));
    NET  = TO - FROM;

    names = string(names(:));
    Ttbl = table(names, FROM, TO, NET, 'VariableNames', {'Series','FROM','TO','NET'});
    OUT = struct('Theta',Theta, 'TSI',TSI, 'TO',TO, 'FROM',FROM, 'NET',NET, 'table',Ttbl);
end
