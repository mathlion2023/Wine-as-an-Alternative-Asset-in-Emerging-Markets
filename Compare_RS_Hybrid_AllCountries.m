%% Compare_RS_Hybrid_AllCountries.m (2026-09-03)
% Replaces Index_Comparizon.m (which read the orphaned, never-produced
% Repeatsales.mat/Projectdata.mat, and picked a country by a hardcoded
% id=1..4 with no documented column-to-country mapping -- exactly the
% kind of positional ambiguity that let the old IndexConstruction.m's
% Australia/Argentina mislabeling go unnoticed). Reads the new, explicitly
% named RS_Hybrid_Comparison_18.mat/csv (built by
% Build_RS_Hybrid_AllCountries.m) instead, and produces the manuscript's
% Table 3 (RS vs Hybrid fit/return statistics, all four countries) and
% Figure 1 (levels/gap/rolling-correlation panels, one column per country)
% on the extended Jan2013-Nov2024 window.
clear; clc;
load('RS_Hybrid_Comparison_18.mat', 'T', 'results', 'countries');

keys = string({countries.key});
labels = ["South Africa","Australia","Argentina","Chile"];
keyOrder = ["SAFW","AUSFW","ARGFW","CHIFW"]; % display order matches Table 1/1b convention

rows = table('Size',[0 10], ...
    'VariableTypes', repmat("double",1,10), ...
    'VariableNames', ["RMSE","MaxGap","CorrRet","RS_AnnMean","RS_AnnVol","RS_Sharpe", ...
                       "HY_AnnMean","HY_AnnVol","HY_Sharpe","N_overlap"]);
countryNames = strings(0,1);
adfP = []; dmStat = []; dmP = [];

for k = 1:numel(keyOrder)
    key = keyOrder(k);
    logRS = log(T.(key+"_RS"));
    logHY = log(T.(key+"_HY"));
    keep = isfinite(logRS) & isfinite(logHY);
    lr = logRS(keep); lh = logHY(keep);
    dRS = diff(lr); dHY = diff(lh);

    rmse   = sqrt(mean((lr - lh).^2));
    maxgap = max(abs(lr - lh));
    corrRet = corr(dRS, dHY, 'Rows','complete');

    statf = @(x) [mean(x)*12, std(x)*sqrt(12), (mean(x)*12)/(std(x)*sqrt(12))];
    sRS = statf(dRS); sHY = statf(dHY);

    rows = [rows; {rmse, maxgap, corrRet, sRS(1), sRS(2), sRS(3), sHY(1), sHY(2), sHY(3), sum(keep)}]; %#ok<AGROW>
    countryNames(end+1,1) = labels(keys==key); %#ok<AGROW>

    % Engle-Granger cointegration: logRS_t = a + b*logHY_t + u_t; ADF on u_t
    X = [ones(numel(lh),1), lh];
    ab = X \ lr; u = lr - X*ab;
    try
        [~, p] = adftest(u, 'Model','ARD', 'Lags', 2);
    catch
        p = NaN;
    end
    adfP(end+1,1) = p; %#ok<AGROW>

    % Diebold-Mariano on AR(1) 1-step forecast errors
    Tn = numel(dRS);
    y1 = dRS(2:end); X1 = [ones(Tn-1,1), dRS(1:end-1)];
    b1 = X1 \ y1; e1 = y1 - X1*b1;
    y2 = dHY(2:end); X2 = [ones(Tn-1,1), dHY(1:end-1)];
    b2 = X2 \ y2; e2 = y2 - X2*b2;
    d = abs(e1) - abs(e2);
    dm = mean(d) / (std(d)/sqrt(numel(d)));
    dmStat(end+1,1) = dm; %#ok<AGROW>
    dmP(end+1,1) = 2*(1-normcdf(abs(dm))); %#ok<AGROW>
end

rows.Country = countryNames;
rows.ADF_p = adfP;
rows.DM_stat = dmStat;
rows.DM_p = dmP;
rows = movevars(rows, 'Country', 'Before', 1);

disp(rows);
writetable(rows, 'tab_rsvhybrid_18.csv');
fprintf('Saved tab_rsvhybrid_18.csv\n');

%% ---- Load the constructed SA monthly proxy (2013-2024), Fig 1 context only ----
% South Africa has no publicly available monthly/quarterly fine-wine price
% index (Wine-Searcher, Strauss & Co, Izak all approached, none available);
% SA_Wine_Price_Index_Monthly_2013-2024.xlsx documents a model-based fallback:
% SAWIS annual wine PPI (Table 7.1) disaggregated to monthly via a national
% wine-CPI indicator (Denton-Cholette). This is NOT transaction data and is
% NOT used in Table 3's fit statistics (RMSE/MaxGap/Corr/N, all still computed
% on the genuine Strauss & Co. RS/Hybrid series from June 2019 only) -- it is
% shown solely as pre-Jun2019 visual/temporal context in Figure 1, replacing
% the earlier Argentina-index placeholder used for the same purpose.
Tsa2 = readtable('SA_Wine_Price_Index_Monthly_2013-2024.xlsx', 'Sheet', 'Monthly SA Wine Price Index');
saPeriod = dateshift(datetime(string(Tsa2.Date), 'InputFormat', 'MMM-yyyy'), 'start', 'month');
saIndexRaw = Tsa2.(Tsa2.Properties.VariableNames{4});

%% ---- Figure 1: levels / gap / rolling 36m corr, one column per country ----
% All four panels share the same 2013-2024 x-axis; South Africa's own
% RS/Hybrid series only exist from June 2019 (Strauss & Co. was founded
% then; Build_RS_Hybrid_AllCountries.m leaves SAFW_RS/SAFW_HY as NaN before
% that, not extrapolated).
f1 = figure('Position',[50 50 1400 900]);
saStart = dateshift(results.SAFW.nativeStart, 'start', 'month');
[tf, loc] = ismember(datenum(T.Period), datenum(saPeriod));
saOnGrid = nan(height(T),1); saOnGrid(tf) = saIndexRaw(loc(tf));
[~, iSplice] = min(abs(datenum(T.Period) - datenum(saStart)));
saScale = 100 / saOnGrid(iSplice);   % splice to HY's own base=100 level at nativeStart
saProxyScaled = saOnGrid * saScale;
for k = 1:numel(keyOrder)
    key = keyOrder(k);
    RS = T.(key+"_RS"); HY = T.(key+"_HY");
    logRS = log(RS); logHY = log(HY);

    subplot(3,4,k);
    plot(T.Period, RS, '-', 'LineWidth',1.1); hold on;
    plot(T.Period, HY, '--', 'LineWidth',1.1);
    if k==1
        preMask = T.Period < saStart;
        saRef = nan(height(T),1);
        saRef(preMask) = saProxyScaled(preMask);
        plot(T.Period, saRef, ':', 'Color', [0.55 0.55 0.55], 'LineWidth',1.3);
        xline(saStart, ':', 'Color', [0.4 0.4 0.4], 'HandleVisibility','off');
    end
    grid on; title(labels(keys==key)); ylabel('Index (base=100)');
    if k==1, legend({'Repeat-Sales','Hybrid','SA proxy (SAWIS PPI+CPI, pre-Jun2019)'}, 'Location','best','FontSize',6); end

    subplot(3,4,4+k);
    plot(T.Period, 100*(logRS-logHY), 'LineWidth',1.0); grid on;
    ylabel('\Delta log(levels) \times 100'); yline(0,'k:');

    subplot(3,4,8+k);
    w = 36;
    dRS = diff(logRS); dHY = diff(logHY);
    n = numel(dRS);
    rc = nan(n,1);
    for t = w:n
        rc(t) = corr(dRS(t-w+1:t), dHY(t-w+1:t), 'Rows','complete');
    end
    plot(T.Period(2:end), rc, 'LineWidth',1.0); grid on;
    ylabel('Rolling 36m corr'); xlabel('Date'); ylim([-1 1]);
end
sgtitle('RS vs. Hybrid: levels, log-level gaps, and rolling 36-month return correlations (2013--2024)');
exportgraphics(f1, 'fig1_RSvHybrid_18.png', 'Resolution', 200);
fprintf('Saved fig1_RSvHybrid_18.png\n');
