function OUT = realizedVar30(X, varargin)
% REALIZEDVAR30  Compute 30-day realized variance/volatility.
%
% USAGE:
%   OUT = realizedVar30(prices,  'Input','prices',  'Annualize',true,  'DoNonOverlap',false);
%   OUT = realizedVar30(returns, 'Input','returns', 'Annualize',true);
%
% INPUT:
%   X : T×N numeric (prices or returns).
%
% Name-Value options:
%   'Input'        : 'prices' (default) or 'returns'
%   'LogReturns'   : true/false, if Input='prices' (default: true)
%   'Annualize'    : true/false (default: true)
%   'TradingDays'  : trading days per year (default: 252)
%   'DoNonOverlap' : also compute non-overlapping 30-day blocks (default: false)
%   'Dates'        : T×1 datetime (optional) to return timetables
%
% OUTPUT struct:
%   OUT.rv30_roll     : T×N rolling 30-day realized variance (overlapping; ends at t)
%   OUT.rv30a_roll    : T×N annualized RV (if Annualize)
%   OUT.rvol30a_roll  : T×N annualized realized volatility = sqrt(OUT.rv30a_roll)
%   OUT.rv30_blocks   : K×N non-overlapping 30-day realized variance (annualized if chosen)
%   OUT.tables        : timetables (if Dates provided)

% -------- options --------
p = inputParser;
addParameter(p, 'Input', 'prices');
addParameter(p, 'LogReturns', true, @(x)islogical(x)||ismember(x,[0,1]));
addParameter(p, 'Annualize', true, @(x)islogical(x)||ismember(x,[0,1]));
addParameter(p, 'TradingDays', 252, @(x)isnumeric(x)&&isscalar(x)&&x>0);
addParameter(p, 'DoNonOverlap', false, @(x)islogical(x)||ismember(x,[0,1]));
addParameter(p, 'Dates', [], @(x)isdatetime(x)||isempty(x));
parse(p, varargin{:});
opt = p.Results;

X = double(X);
[T,N] = size(X);

% -------- returns --------
switch lower(opt.Input)
    case 'prices'
        if opt.LogReturns
            R = diff(log(X));          % log returns
        else
            R = diff(X)./X(1:end-1,:); % simple returns
        end
        R = [nan(1,N); R];            % align back to T
    case 'returns'
        R = X;
    otherwise
        error('Input must be "prices" or "returns".');
end

r2 = R.^2;

% -------- rolling (overlapping) 30-day RV (window t-29..t) --------
rv30_roll = movsum(r2, [29 0], 1, 'omitnan');

% Annualize (variance scales linearly with time)
if opt.Annualize
    rv30a_roll   = rv30_roll * (opt.TradingDays / 30);
    rvol30a_roll = sqrt(rv30a_roll);
else
    rv30a_roll   = [];
    rvol30a_roll = [];
end

% -------- non-overlapping blocks (optional) --------
rv30_blocks = [];
if opt.DoNonOverlap
    keep = floor(T/30)*30;
    if keep == 0
        rv30_blocks = NaN(0,N);
    else
        Rb = R(T-keep+1:T, :);                 % last multiple-of-30 chunk
        r2b = Rb.^2;
        rv30_blocks = squeeze(sum(reshape(r2b, 30, [], N), 1));  % K×N
        if opt.Annualize
            rv30_blocks = rv30_blocks * (opt.TradingDays / 30);
        end
    end
end

% -------- package --------
OUT = struct();
OUT.rv30_roll     = rv30_roll;
OUT.rv30a_roll    = rv30a_roll;
OUT.rvol30a_roll  = rvol30a_roll;
OUT.rv30_blocks   = rv30_blocks;

% -------- optional dated timetables --------
if ~isempty(opt.Dates) && numel(opt.Dates)==T
    D = opt.Dates(:);
    OUT.tables = struct();
    OUT.tables.rv30_roll = array2timetable(rv30_roll, 'RowTimes', D);
    if ~isempty(rv30a_roll)
        OUT.tables.rv30a_roll   = array2timetable(rv30a_roll,   'RowTimes', D);
        OUT.tables.rvol30a_roll = array2timetable(rvol30a_roll, 'RowTimes', D);
    end
    if ~isempty(rv30_blocks)
        K = size(rv30_blocks,1);
        % end-of-block dates (every 30th obs in the last keep segment)
        Dblocks = D(end - K*30 + 30 : 30 : end);
        OUT.tables.rv30_blocks = array2timetable(rv30_blocks, 'RowTimes', Dblocks);
    end
else
    OUT.tables = [];
end
end
