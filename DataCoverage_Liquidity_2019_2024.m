%% DataCoverage_Liquidity_2019_2024.m
% ---------------------------------------------------------------
% Data Coverage & Liquidity Screens (2019–2024)
% Source: "Strauss Wine All updated.xlsx"
% Output:
%   - summary table in MATLAB
%   - CSV: tab_coverage.csv
%   - LaTeX: tab_coverage.tex
% ---------------------------------------------------------------

clear; clc;

%% USER SETTINGS
% FILE = 'Strauss Wine All updated.xlsx';   % <-- exact filename
% SHEET = 'SAOnly';   % or 'Sheet1' if needed
FILE = 'Data Australia';   % <-- exact filename
SHEET = 1;  
DATE_START = datetime(2019,1,1);
DATE_END   = datetime(2024,12,31);
MIN_TRADES_PER_MONTH = 8;                 % change to your threshold
APPEND_FORMAT_TO_ID = false;              % set true if you want ID = WineName|Vintage|Format

%% READ + MAP COLUMNS (robust to header variants)
T0 = readtable(FILE,'Sheet',SHEET);
 col.Country  = pickVar(T0, ["wine_country","Country","country","COUNTRY","Origin"]);

col.Name     = pickVar(T0, ["WineName","name","Label","Wine","Product"]);
col.Vintage  = pickVar(T0, ["Vintage","vintage","Year"]);
col.Date     = pickVar(T0, ["DateSold","SaleDate","Date","ActionDay","AuctionDate","SettlementDate","ym"]);
% price is not used for coverage metrics but we keep for sanity
col.Price    = pickVar(T0, ["PricePer750ml","Price_after_fees","pricePerBottle","Price","AllInPrice","HammerPrice"]);
% optional size/format column
col.Format   = pickVar(T0, ["Format","Bottle","BottleSize","Size","SizeML","ML","Volume"], true);  % optional

T = table;
 T.Country = lower(strtrim(string(T0.(col.Country))));
T.Name    = strtrim(string(T0.(col.Name)));
T.Vintage = double(T0.(col.Vintage));
T.Price   = double(T0.(col.Price));

% Parse dates (handles '01-May-2022', '2019-06-01', Excel serials)
T.Month = normalizeMonth(T0.(col.Date));  % month start (datetime)

% Keep valid rows and window
 valid = ~ismissing(T.Country) & ~ismissing(T.Name) & ~isnan(T.Vintage) ...
         & ~isnat(T.Month) & T.Month>=DATE_START & T.Month<=DATE_END;

% valid = ~ismissing(T.Name) & ~isnan(T.Vintage) ...
%         & ~isnat(T.Month) & T.Month>=DATE_START & T.Month<=DATE_END;
T = T(valid, :);

% Build ItemID: WineName|Vintage (optionally add Format)
if ~isempty(col.Format) && APPEND_FORMAT_TO_ID
    fmt = lower(strtrim(string(T0.(col.Format))));
    T.ItemID = lower(strtrim(T.Name)) + "|" + string(T.Vintage) + "|" + fmt(valid);
else
    T.ItemID = lower(strtrim(T.Name)) + "|" + string(T.Vintage);
end

% Sort canonical
 T = sortrows(T, {'Country','ItemID','Month'});


%% PER-COUNTRY COVERAGE + RS STATS
countries = unique(T.Country);
summary = table('Size',[0 6], ...
    'VariableTypes',["string","double","double","double","double","double"], ...
    'VariableNames',["Country","Trades","UniqueItems","RSPairs","MedianHoldingMonths","MonthsGE_MinTrades"]);

for c = countries.'
    Tc = T(strcmp(T.Country,c), :);

    % --- TRADES / UNIQUE ITEMS ---
    trades      = height(Tc);
    uniqueItems = numel(unique(Tc.ItemID));

    % --- REPEAT-SALES consecutive pairs + holding months ---
    [rsPairs, medHold] = repeatSalesStats(Tc.ItemID, Tc.Month);

    % --- MONTHS WITH >= MIN_TRADES ---
    monthsGE = monthsMeetingThreshold(Tc.Month, MIN_TRADES_PER_MONTH, DATE_START, DATE_END);

    summary = [summary; {string(c), trades, uniqueItems, rsPairs, medHold, monthsGE}]; %#ok<AGROW>
end

% Sort alphabetically (optional)
summary = sortrows(summary, "Country");

%% DISPLAY + SAVE
disp('=== Data Coverage & Liquidity Screens (2019–2024) ===');
disp(summary);

writetable(summary, 'tab_coverage.csv');
writeLatexCoverage(summary, MIN_TRADES_PER_MONTH, 'tab_coverage.tex');

%% --------------- FUNCTIONS ----------------

function name = pickVar(T, candidates, optional)
% Pick first matching column (case-insensitive). If optional==true and none found, return "".
    if nargin < 3, optional = false; end
    vars = T.Properties.VariableNames;
    name = "";
    for c = candidates
        hit = strcmpi(vars, string(c));
        if any(hit), name = vars{hit}; return; end
    end
    if ~optional
        error('Required column not found. Tried: %s', strjoin(string(candidates), ', '));
    end
end

function M = normalizeMonth(x)
% Convert to datetime month starts from a variety of formats.
    if isdatetime(x)
        d = x;
    elseif isnumeric(x)
        % Excel serial or datenum-like
        try
            d = datetime(x, 'ConvertFrom','excel');
        catch
            d = datetime(1899,12,30) + days(x);
        end
    else
        sx = string(x);
        % try common patterns
        tried = false;
        fmts = {'dd-MMM-yyyy','yyyy-MM-dd','dd/MM/yyyy','MM/dd/yyyy','yyyy-MM','MMM yyyy'};
        for k=1:numel(fmts)
            try
                d = datetime(sx, 'InputFormat', fmts{k});
                tried = true; break;
            catch
            end
        end
        if ~tried
            d = datetime(sx); % let MATLAB infer
        end
    end
    d.TimeZone = '';
    M = dateshift(d, 'start','month');
end

function [rsPairs, medHoldMonths] = repeatSalesStats(ItemID, Month)
% Consecutive repeat-sales pairs per item; median holding period in months.
    if isempty(ItemID)
        rsPairs = 0; medHoldMonths = NaN; return;
    end
    [G,~] = findgroups(ItemID);
    rsPairs = 0; diffs = [];
    for g = unique(G).'
        idx = find(G==g);
        if numel(idx) >= 2
            m1 = Month(idx(1:end-1));
            m2 = Month(idx(2:end));
            dt = calmonths(between(m1, m2, 'months'));
            rsPairs = rsPairs + numel(dt);
            diffs = [diffs; dt(:)]; %#ok<AGROW>
        end
    end
    if isempty(diffs)
        medHoldMonths = NaN;
    else
        medHoldMonths = median(diffs, 'omitnan');
    end
end

function monthsGE = monthsMeetingThreshold(Month, minTrades, dateStart, dateEnd)
% Count how many calendar months in [dateStart, dateEnd] have at least minTrades trades.

    if isempty(Month)
        monthsGE = 0; return;
    end

    % Snap to month start and make a full month grid for 2019–2024
    ym = dateshift(Month, 'start','month');
    allMonths = (dateshift(dateStart,'start','month'):calmonths(1):dateshift(dateEnd,'start','month'))';

    % Count trades per observed month (robust to MATLAB version)
    [uniqMonths, ~, grp] = unique(ym);
    counts = accumarray(grp, 1);                 % occurrences per uniq month

    % Align counts to the full grid
    [tf, loc] = ismember(allMonths, uniqMonths);
    countsOnGrid = zeros(numel(allMonths),1);
    countsOnGrid(tf) = counts(loc(tf));

    % Number of months meeting the threshold
    monthsGE = sum(countsOnGrid >= minTrades);
end


function writeLatexCoverage(summary, minTrades, outFile)
% LaTeX table: Data Coverage and Liquidity Screens (2019–2024)
    fid = fopen(outFile,'w'); assert(fid>0, 'Cannot open %s', outFile);
    fprintf(fid, '%% Auto-generated coverage table (2019--2024)\n');
    fprintf(fid, '\\begin{table}[t]\\centering\n');
    fprintf(fid, '\\caption{Data Coverage and Liquidity Screens (2019--2024)}\\label{tab:coverage}\n');
    fprintf(fid, '\\begin{tabular}{lrrrrr}\\toprule\n');
    fprintf(fid, ' & Trades & Unique Items & RS Pairs & Median Holding (mo.) & Months with $\\ge$ %d Trades \\\\\\midrule\n', minTrades);
    for i=1:height(summary)
        fprintf(fid, '%s & %d & %d & %d & %.1f & %d \\\\\n', ...
            summary.Country(i), summary.Trades(i), summary.UniqueItems(i), ...
            summary.RSPairs(i), summary.MedianHoldingMonths(i), summary.MonthsGE_MinTrades(i));
    end
    fprintf(fid, '\\bottomrule\\end{tabular}\n');
    fprintf(fid, '\\begin{flushleft}\\footnotesize\n');
    fprintf(fid, '\\emph{Notes:} RS Pairs are consecutive sales of the same item (WineName$\\times$Vintage% s). ', '');
    fprintf(fid, 'Median holding is the median months between consecutive sales. ');
    fprintf(fid, 'Months with $\\ge$ threshold counted over Jan 2019--Dec 2024.\\end{flushleft}\n');
    fprintf(fid, '\\end{table}\n');
    fclose(fid);
end
