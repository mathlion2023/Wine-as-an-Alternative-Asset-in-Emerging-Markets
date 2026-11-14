%% === AUSTRALIA: one-file run (Data Australia.xlsx) ===
clear; clc;

FILE = "Data Australia.xlsx";    % your uploaded file
SHEET = "Sheet1";                 % change if needed
MIN_TRADES_PER_MONTH = 8;
dateStart = datetime(2002,1,1);
dateEnd   = datetime(2024,12,31);

% Read and normalize
Traw = readtable(FILE,'Sheet',SHEET);
T = normalizeAuctionTable_AUS(Traw);
%%
% Window + sort
T = T(T.TradeDate >= dateStart & T.TradeDate <= dateEnd, :);
T = sortrows(T, {'ItemID','TradeDate'});

% Metrics
trades = height(T);
uniqueItems = numel(unique(T.ItemID));
[rsPairs, medHoldMonths] = repeatSalesStats(T);
monthsGE = monthsMeetingThreshold(T, MIN_TRADES_PER_MONTH, dateStart, dateEnd);

% Show & save LaTeX row (copy into master table later)
fprintf('Australia: Trades=%d | UniqueItems=%d | RSPairs=%d | MedianHold=%.1f | Months>=Min=%d\n', ...
    trades, uniqueItems, rsPairs, medHoldMonths, monthsGE);

% Write a single-row LaTeX table (you can append others later)
S = table("Australia", trades, uniqueItems, rsPairs, medHoldMonths, monthsGE, ...
    'VariableNames', ["Country","Trades","UniqueItems","RSPairs","MedianHoldingMonths","MonthsGE_MinTrades"]);
writeLatexCoverage(S, MIN_TRADES_PER_MONTH, 'tab_coverage_Australia.tex');

%% ---------- Helper: build a composite ItemID for Australia ----------
function T = normalizeAuctionTable_AUS(T0)
% Map your Aussie sheet columns to a standard schema: ItemID | TradeDate | Price
% Tries common names. Edit the 'pick' calls if your headers differ.

    % --- try common header names (case-insensitive) ---
    Producer   = pick(T0, ["Producer","Estate","Winery","Maker"]);
    Label      = pick(T0, ["Label","Wine","Cuvee","Name"]);
    Vintage    = pick(T0, ["Vintage","Year"]);
    Format     = pick(T0, ["Format","BottleFormat","Bottle","Size"]);
    BottleSize = pick(T0, ["BottleSize","SizeML","ML","VolumeML"]);
    LotID      = pick(T0, ["LotID","Lot","SKU","Code","ItemID"]);
    DateCol    = pick(T0, ["TradeDate","SaleDate","AuctionDate","Date","SettlementDate"]);
    PriceCol   = pick(T0, ["AllInPrice","FinalPrice","HammerPrice","Price","Amount","Value"]);

    % --- fallbacks for Format/Size ---
    if isempty(Format) && ~isempty(BottleSize)
        Format = BottleSize;
    end

    % --- compose an ItemID (stable across repeats of the same exact item) ---
    P  = normstr(T0.(Producer));
    L  = normstr(T0.(Label));
    V  = normstr(T0.(Vintage));
    F  = normstr(T0.(Format));
    LID = normstr(maybe(T0, LotID, repmat("",height(T0),1)));

    % Typical composite: Producer|Label|Vintage|Format (plus LotID if you want higher granularity)
    ItemID = string(P) + "|" + string(L) + "|" + string(V) + "|" + string(F);
    % If your LotID is persistent across repeat sales, you can append it:
    % ItemID = ItemID + "|" + string(LID);

    % --- normalize date & price ---
    TradeDate = normalizeDate(T0.(DateCol));
    Price = double(T0.(PriceCol));

    % --- build output ---
    T = table(string(ItemID), TradeDate, Price, 'VariableNames', {'ItemID','TradeDate','Price'});

    % Drop missing dates
    T = T(~isnat(T.TradeDate), :);
end

%% ---------- Shared utilities (same as earlier post) ----------
function name = pick(T, candidates)
    name = "";
    for c = candidates
        if any(strcmpi(c, T.Properties.VariableNames))
            name = string(c); return;
        end
    end
end

function s = normstr(x)
    if iscellstr(x) || isstring(x), s = string(x);
    elseif iscategorical(x),        s = string(x);
    elseif isnumeric(x),            s = string(x);
    else,                           s = string(x);
    end
    s = strtrim(s);
end

function y = maybe(T, name, fallback)
    if isempty(name), y = fallback; else, y = T.(name); end
end

function d = normalizeDate(x)
    if isa(x,'datetime'), d = x;
    elseif isnumeric(x)
        try, d = datetime(x,'ConvertFrom','excel');
        catch, d = datetime(1899,12,30) + days(x);
        end
    else
        try, d = datetime(string(x),'InputFormat','yyyy-MM-dd');
        catch, try, d = datetime(string(x),'InputFormat','dd/MM/yyyy');
        catch, d = datetime(string(x));
        end, end
    end
    d.TimeZone = '';
end

function [rsPairs, medHoldMonths] = repeatSalesStats(T)
    if isempty(T), rsPairs=0; medHoldMonths=NaN; return; end
    G = findgroups(T.ItemID);
    rsPairs = 0; diffsMonths = [];
    for g = unique(G).'
        idx = (G==g); dt = T.TradeDate(idx);
        if numel(dt)>=2
            rsPairs = rsPairs + (numel(dt)-1);
            mm = months(between(dt(1:end-1), dt(2:end), 'months'));
            diffsMonths = [diffsMonths; mm(:)]; %#ok<AGROW>
        end
    end
    medHoldMonths = median(diffsMonths,'omitnan');
end

function monthsGE = monthsMeetingThreshold(T, minTrades, dateStart, dateEnd)
    if isempty(T), monthsGE=0; return; end
    ym = dateshift(T.TradeDate,'start','month');
    G = groupsummary(table(ym), 'ym', 'numel');
    allMonths = (dateshift(dateStart,'start','month'):calmonths(1):dateshift(dateEnd,'start','month'))';
    Gfull = table(allMonths,'VariableNames',{'ym'});
    G = outerjoin(Gfull, G, 'Keys','ym','MergeKeys',true);
    G.numel(isnan(G.numel)) = 0;
    monthsGE = sum(G.numel >= minTrades);
end

function writeLatexCoverage(summary, minTrades, outFile)
    fid = fopen(outFile,'w'); assert(fid>0);
    fprintf(fid, '%% Auto-generated coverage table\n');
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
    fprintf(fid, '\\emph{Notes:} RS Pairs count consecutive sales of the same exact item (producer/label/format/ID). ');
    fprintf(fid, 'Median holding computed from calendar-month differences between consecutive sales. ');
    fprintf(fid, 'Months with $\\ge$ threshold computed over 2019--2024.\\end{flushleft}\n');
    fprintf(fid, '\\end{table}\n'); fclose(fid);
end
