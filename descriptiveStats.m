function out = descriptiveStats(X, freq, doAnnualize)
% DESCRIPTIVESTATS  Compute Mean, StdDev, Skew, Kurtosis, AC(1) (+ N_eff)
%   out = descriptiveStats(X, freq, doAnnualize)
%   - X: timetable/table (T x N) or numeric matrix (T x N)
%   - freq: 'daily'|'weekly'|'monthly'|'quarterly'|'annual' or positive scalar (periods/yr)
%           (default: 'monthly')
%   - doAnnualize: true/false to annualize mean & std (default: true)

    % --- defaults for older MATLAB ---
    if nargin < 2 || isempty(freq),       freq = 'monthly';    end
    if nargin < 3 || isempty(doAnnualize), doAnnualize = true; end

    % --- resolve input to numeric matrix + labels ---
    if istimetable(X)
        labels = string(X.Properties.VariableNames);
        R = X.Variables;     % T x N
    elseif istable(X)
        labels = string(X.Properties.VariableNames);
        R = table2array(X);
    else
        R = X;
        labels = "Var" + string(1:size(R,2));
    end
    R = double(R);

    % --- periods per year ---
    freqPY = resolveFreqPerYear_bc(freq, X);

    % --- core stats by column ---
    nVar = size(R,2);
    m    = NaN(1,nVar);
    s    = NaN(1,nVar);
    sk   = NaN(1,nVar);
    kt   = NaN(1,nVar);
    ac1  = NaN(1,nVar);
    neff = NaN(1,nVar);

    for j = 1:nVar
        x = R(:,j);
        x = x(~isnan(x));
        neff(j) = numel(x);
        if neff(j) < 2, continue; end

        m(j)   = mean(x);
        s(j)   = std(x);
        % Older MATLAB: skewness/kurtosis third arg may not accept 'omitnan'
        sk(j)  = skewness(x, 0);   % bias-corrected
        kt(j)  = kurtosis(x, 0);   % returns Fisher kurtosis (normal = 3)
        ac1(j) = acLag1_bc(x);
    end

    % --- annualize mean & std if requested ---
    if doAnnualize
        m = m * freqPY;
        s = s * sqrt(freqPY);
    end

    % --- assemble output table ---
    T = table(labels(:), m(:), s(:), sk(:), kt(:), ac1(:), neff(:), ...
        'VariableNames', {'Series','Mean','StdDev','Skew','Kurtosis','AC1','N_eff'});

    out.table  = T;
    out.freqPY = freqPY;
    out.labels = labels;
    out.Neff   = neff;
end

%% --------- helpers (backward-compatible) ---------
function f = resolveFreqPerYear_bc(freq, X)
    if isnumeric(freq) && isscalar(freq) && freq > 0
        f = freq; return;
    end
    if ischar(freq) || isstring(freq)
        switch lower(char(freq))
            case 'daily',     f = 252;
            case 'weekly',    f = 52;
            case 'monthly',   f = 12;
            case 'quarterly', f = 4;
            case 'annual',    f = 1;
            otherwise
                % Try to infer if timetable available
                if istimetable(X)
                    t = X.Properties.RowTimes;
                    % crude inference from median step:
                    d = median(diff(t));
                    daysStep = days(d);
                    if daysStep > 330,      f = 1;
                    elseif daysStep > 75,   f = 4;
                    elseif daysStep > 20,   f = 12;
                    elseif daysStep > 4,    f = 52;
                    else,                   f = 252;
                    end
                else
                    error('Unknown freq. Use daily/weekly/monthly/quarterly/annual or a numeric scalar.');
                end
        end
    else
        error('Invalid freq argument.');
    end
end

function r = acLag1_bc(x)
    x = x(:);
    n = numel(x);
    if n < 2, r = NaN; return; end
    x = x - mean(x);
    r = (x(2:end)' * x(1:end-1)) / (x' * x);
end
