function write_full_model_comparison(OUT, names, texFname, csvFname)
% WRITE_FULL_MODEL_COMPARISON  Export ALL candidate models (not just the
% winner) per series to LaTeX + CSV -- the manuscript's Table 4.
%
% Uses OUT.all{i}, the struct array of every trial (GARCH/GJR/EGARCH/
% FIGARCH) that fit_auto_vol_models.m already estimates and keeps in
% memory for each series i, but which no prior script persisted to disk.
%
%   write_full_model_comparison(OUT, names, 'tab_allmodels.tex', 'tab_allmodels.csv')

    if iscell(names), names = string(names); end
    N = numel(names);

    rows = table('Size',[0 7], ...
        'VariableTypes', ["string","string","double","double","double","double","double"], ...
        'VariableNames', ["Index","Model","logL","AIC","BIC","d","tstat_d"]);

    for i = 1:N
        if i>numel(OUT.all) || isempty(OUT.all{i})
            continue;
        end
        trials = OUT.all{i};
        for k = 1:numel(trials)
            tr = trials(k);
            d_ = tr.d; if isempty(d_) || ~isfinite(d_), d_ = NaN; end
            t_ = tr.tstat_d; if isempty(t_) || ~isfinite(t_), t_ = NaN; end
            rows = [rows; {names(i), string(tr.name), safeNum(tr.logL), safeNum(tr.AIC), safeNum(tr.BIC), d_, t_}]; %#ok<AGROW>
        end
    end

    writetable(rows, csvFname);

    fid = fopen(texFname,'w'); assert(fid>0, 'Cannot open %s for writing.', texFname);
    fprintf(fid,'\\begin{table}[t]\\centering\\caption{Model selection by BIC with FIGARCH fractional parameter $d$ (t-stat in parentheses), 18-series panel.}\\label{tab:modelsel}\n');
    fprintf(fid,'\\begin{tabular}{l l r r r l}\\toprule\n');
    fprintf(fid,'Index & Model & logL & AIC & BIC & $d$ (t) \\\\\\midrule\n');

    uIdx = unique(rows.Index, 'stable');
    for ii = 1:numel(uIdx)
        sub = rows(rows.Index==uIdx(ii), :);
        [~, bestRow] = min(sub.BIC);
        for k = 1:height(sub)
            dcell = '';
            if strcmpi(sub.Model(k),'FIGARCH') && isfinite(sub.d(k))
                dcell = sprintf('%.3f (%.2f)', sub.d(k), sub.tstat_d(k));
            end
            mdlName = char(sub.Model(k));
            if k==bestRow, mdlName = ['\textbf{' mdlName '}']; end %#ok<AGROW>
            fprintf(fid,'%s & %s & %s & %s & %s & %s \\\\\n', ...
                char(uIdx(ii)), mdlName, numfmt(sub.logL(k),'%.3f'), numfmt(sub.AIC(k),'%.3f'), numfmt(sub.BIC(k),'%.3f'), dcell);
        end
        if ii<numel(uIdx), fprintf(fid,'\\addlinespace\n'); end
    end

    fprintf(fid,'\\bottomrule\\end{tabular}\n');
    fprintf(fid,'\\begin{flushleft}\\footnotesize ');
    fprintf(fid,'\\emph{Notes:} Lower BIC preferred (\\textbf{bold}), before the significance guardrail is applied -- i.e.\\ the bolded row is BIC-best among all four candidates, which is not necessarily the model ultimately selected in Table 6 once FIGARCH is excluded for insignificant $d$. Keep FIGARCH only when $|t(d)|\\ge 1.96$ \\emph{and} residual diagnostics (LB on $z_t$, LB on $z_t^2$, ARCH-LM) pass.');
    fprintf(fid,'\\end{flushleft}\\end{table}\n');
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
