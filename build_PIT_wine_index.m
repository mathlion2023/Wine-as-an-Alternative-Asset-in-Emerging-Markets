function [logHY_grid, logRS_grid] = build_PIT_wine_index(name, vintage, ymDate, price, useNameDummies, cutoffDate, grid)
% build_PIT_wine_index  Point-in-time (no-lookahead) wine price index.
%
% Peer review item 6: the backtest's wine returns were built from an
% index estimated ONCE using the full 2013-2024 (or 2019-2024) sample --
% every historical month's level depends, through the joint hedonic/GLS
% system, on transactions that occur AFTER that month, which is
% look-ahead leakage for an out-of-sample exercise. This function
% reproduces Build_RS_Hybrid_AllCountries.m's per-country construction
% (same fogartyJonesHybridRS call, same interpolate-within-native-range-
% only policy, same no-extrapolation rule) but truncates the lot-level
% data to ymDate <= cutoffDate first, so the returned index uses only
% information that would genuinely have been available as of cutoffDate.
%
% Inputs: name/vintage/ymDate/price are the FULL (untruncated) per-lot
% arrays for one country (already currency-normalized and cleaned exactly
% as in Build_RS_Hybrid_AllCountries.m); this function does the cutoff
% truncation itself so callers can pass the same arrays for every cutoff
% date without re-cleaning.
%
% Outputs: logHY_grid, logRS_grid are log-index values on the full
% monthly `grid`, NaN for any grid month before the country's native
% start, never extrapolated BACKWARD. Grid months after the country's
% last pre-cutoff estimated period (but still <= cutoffDate) are carried
% forward from that last estimate (last-observation-carried-forward),
% NOT left NaN and NOT linearly extrapolated: a genuinely point-in-time
% investor at a sparse-trading country's cutoff would see "the last
% published index level," not a gap, since (unlike the full-sample
% construction) there is no FUTURE trade left to interpolate from. This
% matters in practice -- e.g. Argentina trades so thinly (Section 3.3/
% Table 1a) that a cutoff falling in a stretch with no recent trade would
% otherwise return NaN for the cutoff month itself, which breaks the
% downstream covariance estimation (a single NaN in Rw propagates through
% the whole Ledoit-Wolf shrinkage matrix via cov()).

    keep = ymDate <= cutoffDate;
    name = name(keep); vintage = vintage(keep); ymDate = ymDate(keep); price = price(keep);

    if numel(price) < 2
        logHY_grid = nan(size(grid)); logRS_grid = nan(size(grid));
        return;
    end

    R = fogartyJonesHybridRS(name, vintage, ymDate, log(price), useNameDummies);
    R.Period = dateshift(R.Period, 'start', 'month');

    logHY_grid = interp1(datenum(R.Period), R.LogIndex_HY, datenum(grid), 'linear');
    logRS_grid = interp1(datenum(R.Period), R.LogIndex_RS, datenum(grid), 'linear');

    % LOCF only for grid points still within the cutoff window (<=
    % cutoffDate); points after cutoffDate must remain NaN (that is the
    % whole point of this function), and points before the native start
    % (no prior value to carry) are left NaN by fillmissing('previous')
    % automatically.
    inCutoff = grid <= cutoffDate;
    logHY_grid(inCutoff) = fillmissing(logHY_grid(inCutoff), 'previous');
    logRS_grid(inCutoff) = fillmissing(logRS_grid(inCutoff), 'previous');
end
