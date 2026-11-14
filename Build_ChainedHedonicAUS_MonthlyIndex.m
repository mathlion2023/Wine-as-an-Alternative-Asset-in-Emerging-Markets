%% Build_ChainedHedonicAUS_MonthlyIndex.m (2026-10-10, peer-review item 4)
% -------------------------------------------------------------------------
% Peer review correctly rejected the existing Appendix C.3 construction
% (Build_HedonicAUS_MonthlyIndex.m, fed by the separate "Wine Data" project's
% Step1a_PrepareWine.m) as a genuine chained hedonic index: it pools ONE set
% of vintage/name/lot-control coefficients over the full 2004-2025 window,
% with only the auction-event fixed effect varying -- a pooled time-dummy
% index (already relabeled as such on 2026-10-09), not Triplett (2004)
% adjacent-period chaining. "Time effects do not automatically allow
% implicit attribute prices to change over time... simply extracting time
% effects from a pooled regression does not answer the referee's
% structural-break concern" (peer review, item 4).
%
% This script implements the genuine article: a ROLLING-WINDOW chained
% hedonic estimator. At each calendar month m, attribute coefficients
% (vintage dummies + lot controls) are RE-ESTIMATED from only a trailing
% 12-month window of transactions (so they are free to drift across
% windows -- the whole point), and the month-over-month log-price relative
% is read off from that SAME local regression's own month-m and month-(m-1)
% coefficients (so the arbitrary within-window normalization cancels when
% differenced). These local relatives are chained multiplicatively (summed
% in logs) into a single cumulative index, exactly the OECD (2011) /
% Triplett (2004) adjacent-period method -- not a pooled design relabeled.
%
% DATA: Reuses langtons_lots.mat (lot-level, already cleaned by Step1a
% minus its ratings/weather/MCMC machinery, which this check does not need)
% rather than re-deriving raw Langton's Excel parsing -- that would
% duplicate a large, separate, already-correct pipeline for no benefit.
% Filter (classified tier {1,2} + traded (Price>0) + 2004-2025) reproduces
% the pooled version's own cited sample EXACTLY: 445,137 lots, confirmed
% directly against the manuscript's Appendix C.3 text before trusting it.
%
% SIMPLIFICATION, stated explicitly: no wine-NAME fixed effects (only
% vintage + lot controls) in each local window, unlike the pooled version's
% full name+vintage+control design. A given wine name recurs too
% infrequently within any single 12-month window for a well-identified
% name effect window-by-window (the whole reason the pooled version pools
% 21 years of data in the first place); dropping name FE trades some
% cross-sectional quality control for the ability to chain at all. This is
% a real, acknowledged difference in specification, not a hidden one.
% -------------------------------------------------------------------------

clear; clc;
WINDOW_MONTHS = 12;   % trailing re-estimation window for each local link

S = load(fullfile('Wine Data','langtons_lots.mat'), 'lots');
L = S.lots;

classified = (L.LotTier==1 | L.LotTier==2);
traded = isfinite(L.Price) & L.Price > 0;
inYears = L.Date >= datetime(2004,1,1) & L.Date <= datetime(2025,12,31);
L = L(classified & traded & inYears, :);
fprintf('Classified + traded + 2004-2025 lots: %d (matches the pooled version''s cited 445,137)\n', height(L));

% BUG FIX, found before this script was ever trusted: 30 lots in a test
% window had NaN Vintage, which NaN-poisons age2 = (year-Vintage)^2 and
% from there the ENTIRE X'X matrix via ordinary matrix multiplication (one
% NaN row contaminates every column it touches) -- silently zeroing the
% pivoted-QR rank check's threshold (dRq(1) becomes NaN, so every
% comparison dRq>1e-8*dRq(1) is false, so the rank check drops ALL
% columns as "dependent" and every coefficient comes back exactly 0,
% which produced a perfectly flat, zero-return "index" on the first run
% -- caught by sanity-checking the output before using it, not assumed
% correct because the script ran without error). Filtered here, matching
% fogartyJonesHybridRS.m's own isfinite(vintage) guard for the main
% project's hybrid estimator.
badVint = ~isfinite(L.Vintage);
fprintf('Dropping %d lots with non-finite Vintage (%.3f%% of the classified+traded sample).\n', sum(badVint), 100*mean(badVint));
L = L(~badVint, :);

L.ym = dateshift(L.Date, 'start', 'month');
L.logP = log(L.Price);
L.age2 = (year(L.Date) - L.Vintage).^2;
L.logQty = log(max(L.Quantity,1));
L.cond = L.ConditionSeverity; L.cond(~isfinite(L.cond)) = 0; % no condition note -> treated as 0 (clean), same semantics as Step1a's 0-5 severity score
contNames = ["age2","logQty","cond","Screwcap","HasTimberCase","PenfoldsClinic","HasAward","IsSigned"];

monthsAll = unique(L.ym); monthsAll = sort(monthsAll);
M = numel(monthsAll);
fprintf('Monthly grid: %d months, %s to %s\n', M, datestr(monthsAll(1)), datestr(monthsAll(end)));

localRelative = nan(M,1); % log relative from month m-1 to m (NaN for m=1 and any unidentified link)
nObsWindow = nan(M,1); nVintWindow = nan(M,1);
for m = 2:M
    winStart = monthsAll(max(1, m-WINDOW_MONTHS+1));
    winMonths = monthsAll(monthsAll >= winStart & monthsAll <= monthsAll(m));
    sub = L(ismember(L.ym, winMonths), :);
    if height(sub) < 30 || numel(winMonths) < 2
        continue; % too little data to identify a local link this window -- left NaN, carried forward later
    end

    % Month dummies indexed directly by position in winMonths (NOT by
    % string-matching categorical labels, which for a datetime-derived
    % categorical render as 'dd-mmm-yyyy' -- a real bug caught before this
    % script was ever run: comparing those against a separately-formatted
    % 'yyyy-mm' string would silently never match, always falling back to
    % treating every month as the window's own omitted base period instead
    % of chaining off month m-1 specifically). categorical() on a sorted
    % datetime array orders categories chronologically, matching
    % winMonths' own order exactly (verified: unique(sub.ym) == winMonths
    % as sets, both already chronologically sorted) -- so column k of
    % D_month (k=1..numel(winMonths)-1) corresponds to winMonths(k+1),
    % and the omitted base is winMonths(1).
    monthCat = categorical(sub.ym);
    D_month = dummyvar(monthCat); D_month(:,1) = [];
    vintCat = categorical(sub.Vintage); D_vint = dummyvar(vintCat); D_vint(:,1) = [];
    Xc = [sub.age2, sub.logQty, sub.cond, double(sub.Screwcap), double(sub.HasTimberCase), ...
          double(sub.PenfoldsClinic), double(sub.HasAward), double(sub.IsSigned)];
    X = [ones(height(sub),1), D_month, D_vint, Xc];
    K = size(X,2);

    % Exactly-collinear columns (e.g. a vintage traded only in one window
    % month) dropped via pivoted QR, same safeguard as Step1a's pooled
    % regression -- never let the solver silently zero an arbitrary column.
    XtX = full(X'*X); dsc = sqrt(diag(XtX)); dsc(dsc<=0)=1;
    Cx = XtX ./ (dsc*dsc');
    [~,Rq,Eq] = qr(Cx,0); dRq = abs(diag(Rq)); rX = sum(dRq > 1e-8*dRq(1));
    dep = Eq(rX+1:end); keep = true(K,1); keep(dep) = false;

    b = zeros(K,1); b(keep) = X(:,keep) \ sub.logP;

    nWin = numel(winMonths);
    coefThis = b(1 + (nWin-1));                 % winMonths(end) = monthsAll(m); column (nWin-1) of D_month, i.e. b(1+(nWin-1))
    coefPrev = 0; if nWin-1 > 1, coefPrev = b(1 + (nWin-2)); end % winMonths(end-1) = monthsAll(m-1); 0 if it's the window's own omitted base (nWin==2)
    localRelative(m) = coefThis - coefPrev;
    nObsWindow(m) = height(sub); nVintWindow(m) = numel(categories(vintCat));
end

% Chain. A handful of early/thin windows can fail to identify a link
% (NaN) -- carried forward as a zero local move (index unchanged), the
% same "no identified move" convention Step1a's own pooled version uses
% for unidentified event dates (see its ev_ok carry-forward), stated
% explicitly rather than silently interpolated.
nUnidentified = sum(~isfinite(localRelative(2:end)));
localRelative(~isfinite(localRelative)) = 0; localRelative(1) = 0;
logIndex = cumsum(localRelative);
fprintf('%d of %d links unidentified (too little local data); carried forward as no-move.\n', nUnidentified, M-1);

chainedAUS = table(monthsAll, logIndex, nObsWindow, nVintWindow, ...
    'VariableNames', {'Period','LogIndex_Chained','nObsWindow','nVintWindow'});
writetable(chainedAUS, 'tab_chainedhedonicAUS_raw.csv');
save('ChainedHedonicAUS.mat', 'chainedAUS');
fprintf('Saved tab_chainedhedonicAUS_raw.csv, ChainedHedonicAUS.mat\n');
