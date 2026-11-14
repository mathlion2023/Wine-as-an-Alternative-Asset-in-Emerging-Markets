%% Portfolio_Backtest_PointInTime_4country.m (2026-10-09, peer-review item 6)
% -------------------------------------------------------------------------
% Portfolio_Backtest_4country.m's wine returns come from
% MasterData_USD_2019_2024_4country.mat, which Build_MasterData_USD_v2.m
% builds by slicing (Tshort = T(T.Period>=shortStart,:)) the FULL
% 2013-2024 hybrid index (RS_Hybrid_Comparison_18.mat) -- itself a SINGLE
% joint GLS estimation over the entire native history for every country.
% That means every historical month's wine-index level, even the SHORT
% panel's first (2019-06), was estimated using lot-level transactions
% through November 2024. Rolling portfolio weights built from this are
% not a genuine out-of-sample exercise: the "trailing 36-month" return
% history at ANY decision date already reflects information that would
% not actually have existed at that date.
%
% This script re-runs the SAME backtest (same solvers, same costs, same
% evaluation window; five portfolio constructions as of the 2026-10-10
% Turnoverpen/BL_Tangency removal below) but replaces the
% wine columns' returns with a POINT-IN-TIME reconstruction: at each
% evaluated month's cutoff date, the four wine indices are rebuilt from
% scratch using ONLY lot-level transactions with a sale date on or before
% that cutoff (build_PIT_wine_index.m / fogartyJonesHybridRS.m, i.e. the
% exact same estimator Build_RS_Hybrid_AllCountries.m uses, just on
% truncated data). Two distinct uses of these reconstructions, matching
% how an investor would actually experience this:
%   (a) WEIGHT-SETTING at decision date tau uses the point-in-time-at-tau
%       reconstruction's trailing 36-month return history (consistent,
%       all 36 months re-estimated together using the same cutoff).
%   (b) REALIZED return for month t (whether inside a holding period or
%       used for performance evaluation) uses the point-in-time-at-t
%       ("first available") reconstruction -- not the point-in-time-at-tau
%       one, and not the final, fully-revised one -- since that is the
%       estimate that would actually have been reported once month t's
%       transactions closed. Benchmark (non-wine) columns are untouched:
%       they are directly observed market series with no estimation
%       look-ahead to begin with.
%
% Decision dates are a subset of evaluated months here, so the set of
% distinct cutoffs needed is exactly the evaluated months (decisionIdx(1)
% through T) -- no extra reconstructions needed for decision dates alone.
% A full timing test of Build_RS_Hybrid_AllCountries.m (all 4 countries,
% full 2013-2024 window) took ~16s; truncated/earlier cutoffs are cheaper,
% so looping this over the SHORT panel's ~30 evaluated months is a matter
% of minutes, not hours -- this was confirmed empirically before
% committing to the full reconstruction rather than a relabel-only fix.
% -------------------------------------------------------------------------

clear; clc;
startDate = datetime(2013,1,1);
endDate   = datetime(2024,11,1);

load('MasterData_USD_2019_2024_4country.mat', 'M');
names = M.Properties.VariableNames(2:end);
N = numel(names);
lvl = M{:, names};
R_full = diff(log(lvl));            % T x N monthly log-returns (wine columns will be overwritten per-cutoff below)
dates = M.Period(2:end);            % T x 1, aligned to R_full
grid  = M.Period;                   % T+1 x 1, levels grid (matches build_PIT_wine_index's expected grid)
[T, ~] = size(R_full);

% BUG FIX (2026-10-10, code-review finding #9): find(ismember(...)) returns
% indices sorted by ascending column POSITION in `names`, not in the order
% of the {...} list given to it, while wineRet's columns below are always
% emitted in the fixed wineKeys loop order (w=1:4, SAFW/AUSFW/ARGFW/CHIFW).
% Rc(:,wineIdx)=wineRet then assigned positionally, relying on the two
% orderings coinciding by coincidence (true today, since Build_MasterData_
% USD_v2.m happens to build columns SA,AUS,ARG,CHL) rather than by
% construction -- the same "Liv-ex100 is the last column" positional-
% assumption bug already found and fixed twice elsewhere in this project.
% Fixed by looking up each wineKeys(w)'s column explicitly, in wineKeys'
% own order, so Rc(:,wineIdx(w)) always receives wineRet(:,w) for the
% correct country regardless of names' column order.
wineColNames = {'SA_wine_USD','AUS_wine_USD','ARG_wine_USD','CHL_wine_USD'}; % must align 1:1 with wineKeys below
wineKeys = ["SAFW","AUSFW","ARGFW","CHIFW"];
wineIdx = zeros(1, numel(wineKeys));
for w_ = 1:numel(wineKeys)
    idx_ = find(strcmp(names, wineColNames{w_}));
    assert(isscalar(idx_), 'Wine column %s not found (or not unique) in panel variable names.', wineColNames{w_});
    wineIdx(w_) = idx_;
end
clear w_ idx_

%% ---- Load raw lot-level data ONCE (same as Build_RS_Hybrid_AllCountries.m) ----
Tver = readtable('Data Australia.xlsx','Sheet','Sheet1');
Tver.ymDate = datetime(Tver.ym, 'InputFormat', 'yyyy-MM');

% CURRENCY: Dr. Verdickt confirmed (2026-11-02) this source file is
% uniformly AUD-denominated regardless of auction house or country, so
% every row converts via the shared normalize_lot_currency.m, matching
% Build_RS_Hybrid_AllCountries.m exactly (supersedes the earlier per-house
% EUR/GBP/ZAR inference this script used).
Tver = normalize_lot_currency(Tver);

% South Africa's own ZAR->USD rate, needed below for the Strauss lots
% (normalize_lot_currency.m only covers the Verdickt/Tver file).
optsFX = detectImportOptions('Wine DATA.xlsx', 'Sheet', 'Sheet1');
optsFX.VariableNamesRange = 'A6'; optsFX.DataRange = 'A7';
Tfx = readtable('Wine DATA.xlsx', optsFX);
Tfx.Properties.VariableNames{1} = 'Date';
Tfx.Period = dateshift(datetime(Tfx.Date), 'start', 'month');
fxKeysC = cellstr(datestr(Tfx.Period, 'yyyy-mm'));
usdzarMap = containers.Map(fxKeysC, num2cell(Tfx.USDZAR));

Tsa = readtable('Strauss Wine All updated.xlsx','Sheet','SAOnly');
Tsa.ymDate = dateshift(Tsa.AuctionDate, 'start', 'month');

countrySpec = struct( ...
    'key',   {'SAFW', 'AUSFW', 'ARGFW', 'CHIFW'}, ...
    'source',{'strauss','verdickt','verdickt','verdickt'}, ...
    'filter',{'', 'Australia', 'Argentina', 'Chile'});

lots = struct();
for c = 1:4
    key = countrySpec(c).key;
    if strcmp(countrySpec(c).source, 'verdickt')
        Tc = Tver(strcmp(Tver.wine_country, countrySpec(c).filter), :);
        Tc = Tc(Tc.ymDate >= startDate & Tc.ymDate <= endDate, :);
        name = string(Tc.name); vintage = Tc.vintage; ymDate = Tc.ymDate; priceLocal = Tc.priceUSD;
    else
        Tc = Tsa(Tsa.ymDate >= startDate & Tsa.ymDate <= endDate, :);
        name = string(Tc.WineName); vintage = Tc.Vintage; ymDate = Tc.ymDate; priceLocal = Tc.PricePer750ml; % still ZAR
    end
    good = isfinite(priceLocal) & priceLocal > 0 & ~isnat(ymDate) & isfinite(vintage);
    lots.(key).name = name(good); lots.(key).vintage = vintage(good);
    lots.(key).ymDate = ymDate(good); lots.(key).price = priceLocal(good); % SAFW still ZAR here
    lots.(key).useNameDummies = ~strcmp(key, 'SAFW');
end

%% ---- Settings (identical to Portfolio_Backtest_4country.m) ----
% Turnoverpen and BL_Tangency were REMOVED from this headline backtest on
% 2026-10-10 (peer-review item 9, per author decision): both relied on
% author-unconfirmed placeholder parameters and collapsed to exactly
% equal-weight by construction (see Table 10's note and
% Portfolio_Optimization_4country.m, where their mechanics are still
% demonstrated as an appendix-only methodology check, not a headline
% result). An explicit EqualWeight portfolio is added here instead as the
% naive comparator.
window     = 36; rebalEvery = 3; costRate = 0.005; kappaAnn = 0.02;
lb = zeros(N,1); ub = ones(N,1);
qpopts = optimoptions('quadprog','Display','off');
fmopts = optimoptions('fmincon','Display','off','MaxFunctionEvaluations',2e5,'MaxIterations',5e3,'Algorithm','sqp');
lpopts = optimoptions('linprog','Display','off');
portTypes = ["GMV","Tangency","RiskParity","CVaR","EqualWeight"];
nP = numel(portTypes);

decisionIdx = window:rebalEvery:T;
nDec = numel(decisionIdx);
evalIdx = decisionIdx(1):T;
cutoffSet = unique(evalIdx);   % distinct R-row indices needing a point-in-time rebuild

%% ---- Build point-in-time wine log-index at every needed cutoff ----
% pitWineRet{ci} is a T x 4 matrix: wine log-returns as they would be
% estimated using ONLY data through dates(cutoffSet(ci)).
fprintf('Building %d point-in-time wine index reconstructions (this is the expensive step)...\n', numel(cutoffSet));
pitWineRet = cell(numel(cutoffSet),1);
% USDZAR is a directly observed FX series (no estimation lag), so it is
% NOT truncated by cutoff -- only the hedonic/RS wine index itself has a
% look-ahead concern. Computed once outside the loop (does not depend on
% cutoffDate).
gridKeys = cellstr(datestr(grid, 'yyyy-mm'));
fxz = nan(numel(grid),1);
for g = 1:numel(grid)
    if isKey(usdzarMap, gridKeys{g}), fxz(g) = usdzarMap(gridKeys{g}); end
end
tic;
for ci = 1:numel(cutoffSet)
    cutoffDate = dates(cutoffSet(ci));
    logIdxAll = nan(numel(grid), 4);
    for w = 1:4
        key = wineKeys(w);
        price_w = lots.(key).price;
        if strcmp(key, "SAFW")
            % Convert ZAR->USD at the LOT level is not meaningful (no
            % per-lot FX date granularity beyond month); convert the
            % resulting MONTHLY index level instead, exactly as
            % Build_MasterData_USD_v2.m does for the full-sample index
            % (usd = @(level,fx) level./fx on USDZAR). Handled after the
            % log-index is built below, for consistency.
        end
        [logHY, ~] = build_PIT_wine_index(lots.(key).name, lots.(key).vintage, lots.(key).ymDate, ...
            price_w, lots.(key).useNameDummies, cutoffDate, grid);
        logIdxAll(:,w) = logHY;
    end
    % SAFW: convert ZAR index level to USD via USDZAR before differencing,
    % matching usd(level,fx)=level./fx in Build_MasterData_USD_v2.m.
    saCol = 1; % SAFW is column 1 of wineKeys
    logIdxAll(:,saCol) = logIdxAll(:,saCol) - log(fxz); % level_USD = level_ZAR/fx  => log diff is log(level)-log(fx)

    wineRet = diff(logIdxAll); % T x 4
    Rc = R_full;
    Rc(:, wineIdx) = wineRet;
    pitWineRet{ci} = Rc;
    if mod(ci,5)==0 || ci==numel(cutoffSet)
        fprintf('  [%d/%d] cutoff %s done (%.0fs elapsed)\n', ci, numel(cutoffSet), datestr(cutoffDate,'yyyy-mm'), toc);
    end
end
cutoffPos = nan(T,1); cutoffPos(cutoffSet) = 1:numel(cutoffSet);
save('PIT_WineReturns_4country.mat', 'pitWineRet', 'cutoffSet', 'cutoffPos', 'dates', 'wineIdx');
fprintf('Point-in-time reconstruction complete (%.0fs total). Saved PIT_WineReturns_4country.mat\n', toc);

%% ---- Rolling backtest: rebalanced version, point-in-time wine returns ----
wealthRebal   = ones(T,nP);
turnoverHist  = zeros(nDec,nP);
wCurrent      = ones(N,nP)/N;
wealthRebal(decisionIdx(1)-1,:) = 1;
w0BuyHold = [];

for d = 1:nDec
    tau = decisionIdx(d);
    Rw_tau = pitWineRet{cutoffPos(tau)};     % point-in-time-AT-tau return matrix (all T rows; slice the window)
    win = (tau-window+1):tau;
    Rw  = Rw_tau(win,:);
    mu  = mean(Rw)';
    Sigma = ledoitWolfShrink(Rw);

    for k = 1:nP
        prevW = wCurrent(:,k);
        switch portTypes(k)
            case "GMV",         w = solveGMV(Sigma, N, lb, ub, qpopts);
            case "Tangency",    w = solveTangency(mu, Sigma, N, qpopts);
            case "RiskParity",  w = solveRiskParity(Sigma, N, lb, ub, fmopts);
            case "CVaR",        w = solveCVaR(Rw, N, lb, ub, lpopts);
            case "EqualWeight", w = ones(N,1)/N;
        end
        turnoverHist(d,k) = sum(abs(w - prevW));
        if d == 1, w0BuyHold(:,k) = w; end %#ok<AGROW>
        wCurrent(:,k) = w;
    end

    nextTau = min(tau + rebalEvery, T);
    for k = 1:nP
        wv = wealthRebal(tau,k);
        wv = wv * (1 - costRate*turnoverHist(d,k));
        w  = wCurrent(:,k);
        for t = (tau+1):nextTau
            % REALIZED return for month t uses the point-in-time-AT-t
            % ("first available") reconstruction, not Rw_tau -- this is
            % the key behavioral difference from a naive "just use the
            % decision-date index for the whole holding period" version.
            Rt_pit = pitWineRet{cutoffPos(t)};
            rel = exp(Rt_pit(t,:))';
            grossPort = w' * rel;
            wineCarry = (kappaAnn/12) * sum(w(wineIdx));
            wv = wv * grossPort * (1 - wineCarry);
            wealthRebal(t,k) = wv;
            w = (w .* rel) / (w'*rel);
        end
        wCurrent(:,k) = w;   % carry drifted weight into next quarter's prevW
    end
end

%% ---- Buy&Hold: set once at first decision date, point-in-time realized returns ----
startIdx = decisionIdx(1);
wealthBH = ones(T,nP);
wealthBH(startIdx,:) = 1;
for k = 1:nP
    w = w0BuyHold(:,k);
    for t = (startIdx+1):T
        Rt_pit = pitWineRet{cutoffPos(t)};
        rel = exp(Rt_pit(t,:))';
        grossPort = w' * rel;
        wineCarry = (kappaAnn/12) * sum(w(wineIdx));
        wealthBH(t,k) = wealthBH(t-1,k) * grossPort * (1 - wineCarry);
        w = (w .* rel) / (w'*rel);
    end
end

%% ---- Performance metrics (use each evaluated month's point-in-time return for the summary stats too) ----
evalRet = nan(numel(evalIdx), N);
for ii = 1:numel(evalIdx)
    t = evalIdx(ii);
    Rt_pit = pitWineRet{cutoffPos(t)};
    evalRet(ii,:) = Rt_pit(t,:);
end
summRebal = summarize(wealthRebal(evalIdx,:), evalRet, portTypes, mean(turnoverHist)');
summBH    = summarize(wealthBH(evalIdx,:),    evalRet, portTypes, zeros(nP,1));

disp('==== Point-in-time Table 11: Rebalanced ====');
disp(summRebal);
writetable(summRebal, 'tab_portfolio_rebal_4country_PIT.csv');

disp('==== Point-in-time Table 12: Buy & Hold ====');
disp(summBH);
writetable(summBH, 'tab_portfolio_buyhold_4country_PIT.csv');

fprintf('Backtest window: %s to %s (%d evaluated months, %d rebalances)\n', ...
    datestr(dates(startIdx),'yyyy-mm'), datestr(dates(end),'yyyy-mm'), numel(evalIdx), nDec);

%% ---- Figures 4-5: overwrite with point-in-time wealth paths (same
% filenames as Portfolio_Backtest_4country.m, since these now supersede
% the look-ahead-biased version referenced by the same figure captions).
idxRebal = 100 * wealthRebal(evalIdx,:) ./ wealthRebal(evalIdx(1),:);
idxBH    = 100 * wealthBH(evalIdx,:)    ./ wealthBH(evalIdx(1),:);
evalDates = dates(evalIdx);

f4 = figure('Name','Rebalanced (fixed weights), point-in-time','Position',[100 100 800 500]);
plot(evalDates, idxRebal, 'LineWidth', 1.3);
grid on; datetick('x','yyyy','keeplimits'); xlabel('Date'); ylabel('Index (base = 100)');
legend(cellstr(portTypes), 'Location', 'northwest');
exportgraphics(f4, 'FixedWeight_4country.png', 'Resolution', 200);
fprintf('Saved FixedWeight_4country.png (point-in-time)\n');

f5 = figure('Name','Buy&Hold (drifting weights), point-in-time','Position',[100 100 800 500]);
plot(evalDates, idxBH, 'LineWidth', 1.3);
grid on; datetick('x','yyyy','keeplimits'); xlabel('Date'); ylabel('Index (base = 100)');
legend(cellstr(portTypes), 'Location', 'northwest');
exportgraphics(f5, 'BHWeight_4country.png', 'Resolution', 200);
fprintf('Saved BHWeight_4country.png (point-in-time)\n');

%% ================= Helper functions (identical to Portfolio_Backtest_4country.m) =================

function Sigma = ledoitWolfShrink(R)
    [Tn,N] = size(R);
    S = cov(R);
    sd = sqrt(diag(S));
    Rcorr = S ./ (sd*sd');
    rbar = (sum(Rcorr(:)) - N) / (N*(N-1));
    F = rbar * (sd*sd');
    F(1:N+1:end) = diag(S);
    Xc = R - mean(R);
    piMat = zeros(N);
    for i = 1:N
        for j = 1:N
            piMat(i,j) = mean((Xc(:,i).*Xc(:,j) - S(i,j)).^2);
        end
    end
    piHat = sum(piMat(:));
    gammaHat = sum((F(:) - S(:)).^2);
    shrinkage = max(0, min(1, (piHat/gammaHat) / Tn));
    Sigma = shrinkage*F + (1-shrinkage)*S;
end

function w = solveGMV(Sigma, N, lb, ub, qpopts)
    w = quadprog(2*Sigma, zeros(N,1), [], [], ones(1,N), 1, lb, ub, [], qpopts);
end

function w = solveTangency(mu, Sigma, N, qpopts)
    y0 = ones(N,1)/N;
    y = quadprog(2*Sigma, zeros(N,1), [], [], mu', 1, zeros(N,1), [], y0, qpopts);
    w = y / sum(y);
end

function w = solveRiskParity(Sigma, N, lb, ub, fmopts)
    b = ones(N,1)/N;
    obj = @(w) logERCObjective(w, Sigma, b);
    w0 = ones(N,1)/N;
    w = fmincon(obj, w0, [], [], ones(1,N), 1, lb, ub, [], fmopts);
end

function val = logERCObjective(w, Sigma, b)
    sp = sqrt(w'*Sigma*w);
    RC = max(w .* (Sigma*w) / sp, 1e-12);
    N = numel(w); val = 0;
    for i = 1:N-1
        for j = i+1:N
            val = val + (log(RC(i)) - log(b(i)) - log(RC(j)) + log(b(j)))^2;
        end
    end
end

function w = solveCVaR(R, N, lb, ub, lpopts)
    alpha = 0.95;
    Tn = size(R,1);
    f = [zeros(N,1); 1; ones(Tn,1)/((1-alpha)*Tn)];
    A = [-R, -ones(Tn,1), -eye(Tn)];
    b = zeros(Tn,1);
    Aeq = [ones(1,N), 0, zeros(1,Tn)]; beq = 1;
    lbv = [lb; -Inf; zeros(Tn,1)]; ubv = [ub; Inf; Inf(Tn,1)];
    x = linprog(f, A, b, Aeq, beq, lbv, ubv, lpopts);
    w = x(1:N);
end

function T = summarize(wealth, R, portTypes, avgTurnover)
    nP = size(wealth,2);
    ret = diff(wealth) ./ wealth(1:end-1,:);
    % BUG FIX (2026-10-19, co-author Faff flagged pre-resubmission): see the
    % identical fix/comment in Portfolio_Backtest_PointInTime_18.m -- Appendix
    % A states Sharpe/volatility use log(1+R_{p,t+1}), but this computed them
    % on simple ret directly. ret is kept for CumRet/CVaR/MaxDrawdown.
    logret = log(1 + ret);
    annMean = mean(logret)'*12;
    annVol  = std(logret)'*sqrt(12);
    sharpe  = annMean ./ annVol;
    cvar95  = nan(nP,1);
    for k = 1:nP
        r = sort(ret(:,k));
        m = max(1, round(0.05*numel(r)));
        cvar95(k) = -mean(r(1:m));
    end
    cumRet = wealth(end,:)' - 1;
    maxDD = nan(nP,1);
    for k = 1:nP
        peak = cummax(wealth(:,k));
        dd = wealth(:,k)./peak - 1;
        maxDD(k) = min(dd);
    end
    T = table(portTypes', cumRet, annMean, annVol, sharpe, cvar95, maxDD, avgTurnover, ...
        'VariableNames', {'Portfolio','CumRet','AnnMean','AnnVol','Sharpe','CVaR95','MaxDrawdown','AvgTurnover'});
end
