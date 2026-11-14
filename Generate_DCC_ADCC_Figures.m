%% Generate_DCC_ADCC_Figures.m (2026-09-03)
% Regenerates the manuscript's Figures 2-3 (time-varying DCC/ADCC
% correlations) from the CORRECTED ResidualSTD_4country.mat (0-of-18 FIGARCH,
% post the figarch_nll_gauss.m bug fix) and the already-estimated
% DCC_ADCC_Results_4country.mat (a,b,g from DCC_ADCC_Parameters.m).
%
% No prior script in this repo actually saved these figures to disk (grep
% for DCC.eps/ADCC.eps/saveas/print(/exportgraphics across all .m files
% returns nothing) -- the .eps files present in this folder were exported
% by hand at some earlier point, from the OLD 65-month/9-series pipeline,
% and are now stale. This script both computes AND saves them, on the
% current 143-month/18-series/corrected-residual pipeline.
%
% Plots each series' pairwise correlation with Liv-ex100 (the natural
% benchmark row/column, matching the original manuscript's Figure 2/3
% legend convention of one line per non-benchmark series).
clear; clc;
load('ResidualSTD.mat');           % stdResid (T x 17, LONG panel, no SA), names
load('DCC_ADCC_Results.mat');      % dcc, adcc structs (theta, etc.)

Z = stdResid;
[T, N] = size(Z);
livexIdx = find(strcmpi(names, 'LivEx100_USD'));
assert(~isempty(livexIdx), 'LivEx100_USD not found in names.');
otherIdx = setdiff(1:N, livexIdx);

Qbar = cov(Z, 'partialrows');
Sbar = cov(Z .* (Z < 0), 'partialrows');

Rt_dcc  = computeDCC(Z, Qbar, dcc.theta(1), dcc.theta(2));
Rt_adcc = computeADCC(Z, Qbar, Sbar, adcc.theta(1), adcc.theta(2), adcc.theta(3));

% Try to recover a date axis from the master dataset; fall back to index.
try
    load('MasterData_USD_2013_2024.mat', 'M');
    dates = M.Period(2:end); % aligned to returns/residuals (T obs)
    useDates = numel(dates) == T;
catch
    useDates = false;
end

labelMap = containers.Map( ...
    {'SA_wine_USD','AUS_wine_USD','ARG_wine_USD','CHL_wine_USD','JALSH_USD', ...
     'ASX200_USD','MERVAL_USD','IPSA_USD','CAC_USD','MIB_USD','IBEX_USD', ...
     'PSI_USD','DAX_USD','SPX_USD','MSCIEM_USD','GlobalAgg_USD','Gold_USD'}, ...
    {'SAFW','AUSFW','ARGFW','CHIFW','JALSH','ASX200','MERVAL','IPSA','CAC40', ...
     'FTSE MIB','IBEX35','PSI','DAX','S&P 500','MSCI EM','GlobalAgg','Gold'});

f2 = figure('Name','DCC correlations with Liv-ex100','Position',[100 100 900 550]);
hold on;
for k = 1:numel(otherIdx)
    j = otherIdx(k);
    series = squeeze(Rt_dcc(livexIdx, j, :));
    lbl = string(names(j));
    if isKey(labelMap, char(lbl)), lbl = labelMap(char(lbl)); end
    if useDates
        plot(dates, series, 'LineWidth', 1.1, 'DisplayName', lbl);
    else
        plot(series, 'LineWidth', 1.1, 'DisplayName', lbl);
    end
end
grid on; box on;
if useDates, datetick('x','yyyy','keeplimits'); xlabel('Year'); else, xlabel('t'); end
ylabel('Corr(DCC) with Liv-ex100');
legend('Location','eastoutside','FontSize',7);
exportgraphics(f2, 'DCC_18.png', 'Resolution', 200);
fprintf('Saved DCC_18.png\n');

f3 = figure('Name','ADCC correlations with Liv-ex100','Position',[100 100 900 550]);
hold on;
for k = 1:numel(otherIdx)
    j = otherIdx(k);
    series = squeeze(Rt_adcc(livexIdx, j, :));
    lbl = string(names(j));
    if isKey(labelMap, char(lbl)), lbl = labelMap(char(lbl)); end
    if useDates
        plot(dates, series, 'LineWidth', 1.1, 'DisplayName', lbl);
    else
        plot(series, 'LineWidth', 1.1, 'DisplayName', lbl);
    end
end
grid on; box on;
if useDates, datetick('x','yyyy','keeplimits'); xlabel('Year'); else, xlabel('t'); end
ylabel('Corr(ADCC) with Liv-ex100');
legend('Location','eastoutside','FontSize',7);
exportgraphics(f3, 'ADCC_18.png', 'Resolution', 200);
fprintf('Saved ADCC_18.png\n');

%% ---- local copies of DCC_ADCC_Parameters.m's helper functions ----
% (those are locally-scoped there and not reusable without duplication)
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
