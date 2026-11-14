function [R, se_gamma_hy] = fogartyJonesHybridRS(name, vintage, ymDate, logPrice, useNameDummies)
% Fogarty & Jones (2011) hybrid hedonic-repeat-sales estimator.
% Extracted (2026-10-09) from Build_RS_Hybrid_AllCountries.m into its own
% file so Portfolio_Backtest_PointInTime_*.m (the no-lookahead backtest,
% peer-review item 6) can call the identical estimator on truncated data
% without duplicating it -- the same anti-pattern (two near-identical
% copies drifting apart) already flagged elsewhere in this project for
% fit_tb.m/figarch_nll_gauss.m. Logic is unchanged from the version this
% was extracted from.
%
% useNameDummies (default true): set false when the label dimension is
% too sparse relative to the sample for a well-identified label effect
% (matches Appendix B's already-established treatment of South Africa).
%
% Second output se_gamma_hy (2026-10-09, peer-review item 3): item-
% clustered sandwich standard errors for the hybrid time-dummy
% coefficients g_hy, computed ONLY if requested (nargout>1) since it adds
% real cost and most callers (e.g. the point-in-time backtest, run 100+
% times) only need the point estimate. The manuscript's Section 4.2 had
% claimed "uncertainty quantified via heteroskedasticity-robust (sandwich)
% standard errors and the delta method" for the index, but no script
% anywhere in this repo actually computed any such SE -- this was a real
% gap between the text and the code, not just an ambiguity, found while
% verifying that claim rather than taking it at face value. Clustering is
% by item (name x vintage), since a hedonic observation and every
% repeat-sales pair built from it share the same underlying trade-level
% error (exactly the non-block-diagonal structure Section 4.2 already
% derives analytically) -- so cluster-robust, not plain heteroskedasticity-
% robust, is the right correction given that derivation.

    if nargin < 5, useNameDummies = true; end
    wantSE = nargout > 1;
    ym = string(datestr(ymDate, 'yyyy-mm'));

    %% 2. Aggregate lot sales to one price per (name, vintage, ym) cell
    [G0, aggName, aggVint, aggYm] = findgroups(name, vintage, ym);
    aggLogP  = splitapply(@mean,  logPrice, G0);
    Tagg = table(aggName, aggVint, aggYm, aggLogP, 'VariableNames', {'name','vintage','ym','logP'});
    n = height(Tagg);

    %% 3. Hedonic design matrix Xh = [intercept | (name) | vintage | time]
    timeCat    = categorical(Tagg.ym);
    timeLevels = categories(timeCat);
    nT         = numel(timeLevels);
    timeCode   = double(timeCat);
    D_time     = dummyvar(timeCat); D_time(:,1) = [];

    if useNameDummies
        nameCat = categorical(Tagg.name);
        D_name  = dummyvar(nameCat); D_name(:,1) = [];
    else
        D_name = zeros(n,0);
    end

    vintCat = categorical(string(Tagg.vintage));
    D_vint  = dummyvar(vintCat); D_vint(:,1) = [];

    Xh = [ones(n,1), D_name, D_vint, D_time];
    K  = size(Xh,2);
    startIdx = 1 + size(D_name,2) + size(D_vint,2) + 1;

    %% 4. Repeat-sales pairs: all pairwise period combinations per wine+vintage
    Tagg.ID = strcat(Tagg.name, "__", string(Tagg.vintage));
    G = findgroups(Tagg.ID);
    nGroups = max(G);
    pairEarly = cell(nGroups,1); pairLate = cell(nGroups,1); pairGroup = cell(nGroups,1);
    for g = 1:nGroups
        idx = find(G == g);
        if numel(idx) < 2, continue; end
        [~, ord] = sort(timeCode(idx));
        idxSorted = idx(ord);
        combos = nchoosek(1:numel(idxSorted), 2);
        pairEarly{g} = idxSorted(combos(:,1));
        pairLate{g}  = idxSorted(combos(:,2));
        pairGroup{g} = repmat(g, size(combos,1), 1); % which item cluster this RS pair belongs to
    end
    i1 = cat(1, pairEarly{:}); i2 = cat(1, pairLate{:});
    rsGroup = cat(1, pairGroup{:}); % G-index of each RS row, for clustering below
    J = numel(i1);
    Yrs = Tagg.logP(i2) - Tagg.logP(i1);

    p1 = timeCode(i1); p2 = timeCode(i2);
    rows = (1:J)';
    keepLate = p2 > 1; keepEarly = p1 > 1;
    colLate  = startIdx - 2 + p2; colEarly = startIdx - 2 + p1;
    Drs = sparse([rows(keepLate); rows(keepEarly)], ...
                 [colLate(keepLate); colEarly(keepEarly)], ...
                 [ones(sum(keepLate),1); -ones(sum(keepEarly),1)], J, K);

    %% 5. Variance components + pure RS coefficients (Fogarty & Jones eqs. 5-6)
    Drs_slim = Drs(:, startIdx:end);
    AtA = full(Drs_slim' * Drs_slim);
    Atb = full(Drs_slim' * Yrs);
    % BUG FIX (2026-10-10, code-review finding #8): AtA\Atb had no rank/
    % conditioning check. build_PIT_wine_index.m calls this function ~150
    % times on progressively shrinking truncated samples near a thin
    % country's (e.g. Argentina's) earliest trading history, where AtA can
    % become near-singular; backslash would then silently return a finite
    % but numerically unstable solution with no warning in batch mode. Guard
    % with an explicit rcond check and fall back to the minimum-norm pinv
    % solution (with a warning) when ill-conditioned.
    rc_rs = rcond(AtA);
    if ~isfinite(rc_rs) || rc_rs < 1e-12
        warning('fogartyJonesHybridRS:illConditionedRS', ...
            'Repeat-sales AtA is near-singular (rcond=%.3e, J=%d pairs, K=%d periods) -- falling back to pinv.', rc_rs, J, size(AtA,1));
        b_rs = pinv(AtA) * Atb;
    else
        b_rs = AtA \ Atb;                % <-- this IS the pure RS index (log-diffs)
    end
    res_rs = Yrs - Drs_slim * b_rs;
    Trs = size(Drs_slim, 2);
    dof_rs = max(J - Trs, 1);
    sigma2_e = 0.5 * (sum(res_rs.^2) / dof_rs);

    b_h = Xh \ Tagg.logP;
    res_h = Tagg.logP - Xh*b_h;
    dof_h = max(n - K, 1);
    sigma2_eps = sum(res_h.^2) / dof_h;

    %% 6. Stack hedonic + repeat-sales equations and solve by GLS (hybrid)
    Yall = [Tagg.logP; Yrs];
    Xall = [sparse(Xh); Drs];
    w1 = 1/max(sigma2_eps, eps); w2 = 1/max(2*sigma2_e, eps);
    Wbig = [w1*ones(n,1); w2*ones(J,1)];
    nAll = n + J;
    WX = spdiags(Wbig, 0, nAll, nAll) * Xall;
    XtWX = full(WX' * Xall); XtWY = full(WX' * Yall);
    % BUG FIX (2026-10-10, code-review finding #8): same rcond guard as
    % b_rs above, for the stacked hybrid GLS solve.
    rc_gls = rcond(XtWX);
    if ~isfinite(rc_gls) || rc_gls < 1e-12
        warning('fogartyJonesHybridRS:illConditionedGLS', ...
            'Hybrid GLS XtWX is near-singular (rcond=%.3e, K=%d periods) -- falling back to pinv.', rc_gls, size(XtWX,1));
        b_gls = pinv(XtWX) * XtWY;
    else
        b_gls = XtWX \ XtWY;
    end

    %% 7. Extract time-dummy coefficients -> both indices
    g_hy = [0; b_gls(startIdx : startIdx + (nT-2))];
    g_rs = [0; b_rs];

    %% 7b. Item-clustered sandwich SEs for g_hy (only if requested, see header)
    if wantSE
        resid_all = full(Yall - Xall*b_gls);          % n+J stacked residuals
        clusterID = [G; rsGroup];                      % n+J: item cluster of every row (hedonic rows use G, RS rows use rsGroup)
        XtWXinv = pinv(XtWX);                           % (Z'WZ)^{-1}; pinv rather than inv since Xh (hence XtWX) can be rank-deficient (Section 4.2/Appendix: e.g. Chile)
        uniqClusters = unique(clusterID);
        meat = zeros(K,K);
        WXfull = full(WX);
        for ci = 1:numel(uniqClusters)
            rowsG = (clusterID == uniqClusters(ci));
            s_g = WXfull(rowsG,:)' * resid_all(rowsG);  % K x 1 score contribution of this cluster
            meat = meat + s_g * s_g';
        end
        % Small-sample cluster correction (Cameron & Miller 2015, eq. 11): G/(G-1) * (N-1)/(N-K)
        Gn = numel(uniqClusters); Nn = nAll;
        cdf_adj = (Gn/max(Gn-1,1)) * ((Nn-1)/max(Nn-K,1));
        Vb = cdf_adj * (XtWXinv * meat * XtWXinv);
        se_all = sqrt(max(diag(Vb), 0));                % K x 1, one per parameter (floor at 0 against numerical negative-diagonal noise)
        se_gamma_hy = [0; se_all(startIdx : startIdx + (nT-2))]; % base period has SE=0 by normalization, matching g_hy(1)=0
    else
        se_gamma_hy = [];
    end

    R = table(datetime(timeLevels,'InputFormat','yyyy-MM'), g_hy, g_rs, ...
              'VariableNames', {'Period','LogIndex_HY','LogIndex_RS'});
    R.nCellsUsed = repmat(n, height(R), 1);
    R.nPairsUsed = repmat(J, height(R), 1);
end
