function write_figarch_only_table(OUT, names, texFname, csvFname)
% WRITE_FIGARCH_ONLY_TABLE  Export the FIGARCH(1,d,1) fit for EVERY series
% (not just the ones where it was ultimately selected) -- the manuscript's
% Table 5. Uses OUT.all{i}, which already contains the FIGARCH trial
% regardless of whether the guardrail excluded it from final selection.
%
%   write_figarch_only_table(OUT, names, 'tab_figarch_all.tex', 'tab_figarch_all.csv')

    if iscell(names), names = string(names); end
    N = numel(names);

    rows = table('Size',[0 6], ...
        'VariableTypes', ["string","double","double","double","double","double"], ...
        'VariableNames', ["Index","d","tstat_d","logL","AIC","BIC"]);

    for i = 1:N
        if i>numel(OUT.all) || isempty(OUT.all{i})
            continue;
        end
        trials = OUT.all{i};
        mask = arrayfun(@(s) strcmpi(s.name,'FIGARCH'), trials);
        if ~any(mask), continue; end
        tr = trials(find(mask,1,'first'));
        rows = [rows; {names(i), safeNum(tr.d), safeNum(tr.tstat_d), safeNum(tr.logL), safeNum(tr.AIC), safeNum(tr.BIC)}]; %#ok<AGROW>
    end

    writetable(rows, csvFname);

    fid = fopen(texFname,'w'); assert(fid>0, 'Cannot open %s for writing.', texFname);
    fprintf(fid,'\\begin{table}[t]\\centering\\caption{FIGARCH(1,$d$,1) estimates by index: $d$ (t-stat) and information criteria, 18-series panel.}\\label{tab:figarchonly}\n');
    fprintf(fid,'\\begin{tabular}{lrrrr}\\toprule\n');
    fprintf(fid,'Index & FIGARCH(1,$d$,1) & logL & AIC & BIC \\\\\\midrule\n');
    for i = 1:height(rows)
        sig = isfinite(rows.tstat_d(i)) && abs(rows.tstat_d(i))>=1.96;
        dcell = sprintf('$d=%.3f\\,(%.2f)$', rows.d(i), rows.tstat_d(i));
        if sig, dcell = [dcell '$^{*}$']; end %#ok<AGROW>
        fprintf(fid,'%s & %s & %s & %s & %s \\\\\n', char(rows.Index(i)), dcell, ...
            numfmt(rows.logL(i),'%.3f'), numfmt(rows.AIC(i),'%.3f'), numfmt(rows.BIC(i),'%.3f'));
    end
    fprintf(fid,'\\bottomrule\\end{tabular}\n');
    fprintf(fid,'\\begin{flushleft}\\footnotesize \\emph{Notes:} Gaussian QML with Hessian-based s.e.; $^{*}$ indicates $|t(d)|\\ge 1.96$. This is the FIGARCH fit estimated for every series regardless of whether it was ultimately selected (Table 6) -- most series here fail the significance guardrail and are filtered by GJR-GARCH or EGARCH instead.\\end{flushleft}\\end{table}\n');
    fclose(fid);
end

function v = safeNum(x)
    if isempty(x) || ~isnumeric(x) || ~isscalar(x) || ~isfinite(x)
        v = NaN;
    else
        v = x;
    end
end

function s = numfmt(x, fmt)
    if nargin<2, fmt = '%.3g'; end
    if ~isnumeric(x) || ~isscalar(x) || ~isfinite(x)
        s = '';
    else
        s = sprintf(fmt, x);
    end
end
