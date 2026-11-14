%% =======================================================================
%  Spillover_Analysis_ARGCHLexcluded_4country.m (2026-10-11, peer-review-panel item)
%  -----------------------------------------------------------------------
%  A reviewer asked whether the paper's panel-wide null dependence/
%  connectedness findings (Tables 7, 8) partly reflect pooling in
%  Argentina and Chile, which Section 3.3 already flags as the most
%  selection-exposed and thinly-traded markets in the sample (Table 1a:
%  44% and 13% of months interpolated respectively). This script re-runs
%  the identical small-system connectedness methodology of
%  Spillover_Analysis_SmallSystem_4country.m (same VAR/GFEVD/bootstrap
%  code, byte-identical functions), restricted to the two deeper markets
%  only: South Africa, Australia, Liv-ex 100, S&P 500, Gold (5 series,
%  dropping ARGFW/CHIFW from the existing 7-series small system).
% ========================================================================

clearvars; clc;
load('ResidualSTD_4country.mat')   % stdResid, names
smallNames = ["SA_wine_USD","AUS_wine_USD","LivEx100_USD","SPX_USD","Gold_USD"];
namesAll = string(names(:))';
[~, colIdx] = ismember(smallNames, namesAll);
assert(all(colIdx>0), 'One or more series not found in ResidualSTD_4country.mat');
Z = stdResid(:, colIdx);
names = smallNames;
[T,N] = size(Z);
fprintf('ARG/CHL-excluded spillover: N=%d series, T=%d observations (%.1f obs/parameter at p=1)\n', ...
    N, T, T/N^2);

H    = 10; pMin = 1; pMax = 6; crit = 'BIC';
Z    = Z - mean(Z,'omitnan');

%% RET
p_ret = selectVARlag(Z, pMin, pMax, crit);
fprintf('[RET] Selected VAR order p = %d by %s\n', p_ret, upper(crit));
[VARr, SigU_r] = varEstimate(Z, p_ret);
ThetaH_r = gfevd(VARr, SigU_r, H);
ConnRET  = connectedness(ThetaH_r, names);
disp('=== ARG/CHL-excluded connectedness (RETURNS) ==='); disp(ConnRET.table);
fprintf('[RET] Total Spillover Index = %.3f%%\n', ConnRET.TSI);
writetable(ConnRET.table, 'tab_spillover_RET_argchlexcl_4country.csv');

%% VOL
Z2 = (Z.^2) - mean(Z.^2,'omitnan');
p_vol = selectVARlag(Z2, pMin, pMax, crit);
fprintf('[VOL] Selected VAR order p = %d by %s\n', p_vol, upper(crit));
[VARv, SigU_v] = varEstimate(Z2, p_vol);
ThetaH_v = gfevd(VARv, SigU_v, H);
ConnVOL  = connectedness(ThetaH_v, names);
disp('=== ARG/CHL-excluded connectedness (VOLATILITY) ==='); disp(ConnVOL.table);
fprintf('[VOL] Total Spillover Index = %.3f%%\n', ConnVOL.TSI);
writetable(ConnVOL.table, 'tab_spillover_VOL_argchlexcl_4country.csv');

%% Moving-block bootstrap for NET spillover stability (RET and VOL channels)
rng(42);
Bboot = 500; blockLen = 6;
NETboot_ret = nan(Bboot, N);
NETboot_vol = nan(Bboot, N);
nBlocks = ceil(T/blockLen);
for b = 1:Bboot
    starts = randi(T-blockLen+1, nBlocks, 1);
    idx = [];
    for j = 1:nBlocks
        idx = [idx, starts(j):(starts(j)+blockLen-1)]; %#ok<AGROW>
    end
    idx = idx(1:T);
    Zb  = Z(idx,:);
    try
        pb = selectVARlag(Zb, pMin, pMax, crit);
        [VARb, SigUb] = varEstimate(Zb, pb);
        Thb = gfevd(VARb, SigUb, H);
        Cb  = connectedness(Thb, names);
        NETboot_ret(b,:) = Cb.NET';
    catch
    end
    Z2b = (Zb.^2) - mean(Zb.^2,'omitnan');
    try
        pb2 = selectVARlag(Z2b, pMin, pMax, crit);
        [VARb2, SigUb2] = varEstimate(Z2b, pb2);
        Thb2 = gfevd(VARb2, SigUb2, H);
        Cb2  = connectedness(Thb2, names);
        NETboot_vol(b,:) = Cb2.NET';
    catch
    end
end
lo_ret = prctile(NETboot_ret, 5, 1)'; hi_ret = prctile(NETboot_ret, 95, 1)';
lo_vol = prctile(NETboot_vol, 5, 1)'; hi_vol = prctile(NETboot_vol, 95, 1)';
nOK_ret = sum(~isnan(NETboot_ret(:,1)));
nOK_vol = sum(~isnan(NETboot_vol(:,1)));
fprintf('Bootstrap: %d/%d RET draws and %d/%d VOL draws converged (block=%d months)\n', nOK_ret, Bboot, nOK_vol, Bboot, blockLen);

BootTbl = table(names(:), ConnRET.NET, lo_ret, hi_ret, ConnVOL.NET, lo_vol, hi_vol, ...
    'VariableNames', {'Series','NET_RET','NET_RET_p05','NET_RET_p95','NET_VOL','NET_VOL_p05','NET_VOL_p95'});
disp(BootTbl);
writetable(BootTbl, 'tab_spillover_argchlexcl_bootstrapCI_4country.csv');

%% ====================== FUNCTIONS (identical to Spillover_Analysis_SmallSystem_4country.m) ======================
function p = selectVARlag(Y, pMin, pMax, crit)
    [T,~] = size(Y); best = inf; p = pMin;
    for k = pMin:pMax
        [~, SigU] = varEstimate(Y, k);
        logL = - (T-k)*0.5*( size(Y,2)*log(2*pi) + log(det(SigU)) + size(Y,2) );
        kpars = size(Y,2)^2 * k;
        AIC = -2*logL + 2*kpars; BIC = -2*logL + kpars*log(T-k);
        val = strcmpi(crit,'AIC')*AIC + strcmpi(crit,'BIC')*BIC;
        if val < best, best=val; p=k; end
    end
end

function [VAR, SigU] = varEstimate(Y, p)
    [T,N] = size(Y); X = [];
    for j=1:p, X = [X, Y((p-j+1):(T-j), :)]; end %#ok<AGROW>
    Yt = Y((p+1):T, :); X = [ones(T-p,1), X];
    B = (X' * X) \ (X' * Yt); U = Yt - X * B;
    SigU = (U' * U) / (T - p - (1 + N*p));
    VAR.B = B; VAR.N = N; VAR.p = p; VAR.SigU = SigU;
end

function ThetaH = gfevd(VAR, SigU, H)
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

function OUT = connectedness(Theta, names)
    N = size(Theta,1); %#ok<NASGU>
    offdiagSum = sum(Theta, 'all') - sum(diag(Theta));
    TSI = 100 * offdiagSum / size(Theta,1);
    TO   = 100 * (sum(Theta,1)' - diag(Theta));
    FROM = 100 * (sum(Theta,2) - diag(Theta));
    NET  = TO - FROM;
    names = string(names(:));
    Ttbl = table(names, FROM, TO, NET, 'VariableNames', {'Series','FROM','TO','NET'});
    OUT = struct('Theta',Theta, 'TSI',TSI, 'TO',TO, 'FROM',FROM, 'NET',NET, 'table',Ttbl);
end
