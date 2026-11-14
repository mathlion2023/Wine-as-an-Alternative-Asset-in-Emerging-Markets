function write_latex_model_table(OUT, names, fname)
% WRITE_LATEX_MODEL_TABLE  Export selected model + key params to LaTeX.
% OUT from fit_auto_vol_models; names is N×1 string/cellstr.

    if iscell(names), names = string(names); end
    fid = fopen(fname,'w'); assert(fid>0, 'Cannot open %s for writing.', fname);

    fprintf(fid,'\\begin{table}[t]\\centering\\caption{Selected volatility models and key parameters.}\\label{tab:volsel}\n');
    fprintf(fid,'\\begin{tabular}{l l r r r r r}\\toprule\n');
    fprintf(fid,'Index & Model & $\\omega$ & $\\phi$ & $\\beta$ & $d$ & BIC \\\\\\midrule\n');

    N = numel(names);
    for i = 1:N
        % Skip if nothing selected
        if i>numel(OUT.choice) || isempty(OUT.choice{i})
            continue;
        end

        idxName = aschar(names(i));
        mdl     = aschar(OUT.choice{i});

        % Pull parameters table (may be empty/non-FIGARCH)
        omega = NaN; phi = NaN; beta = NaN; d = NaN;
        if i<=numel(OUT.params) && ~isempty(OUT.params{i})
            PT = OUT.params{i};
            omega = pickParam(PT,'omega',omega);
            phi   = pickParam(PT,'phi',phi);
            beta  = pickParam(PT,'beta',beta);
            d     = pickParam(PT,'d',d);       % will stay NaN for non-FIGARCH
        end

        % BIC
        BIC = NaN;
        if i<=numel(OUT.stats) && ~isempty(OUT.stats{i}) && ismember('BIC', OUT.stats{i}.Properties.VariableNames)
            val = OUT.stats{i}.BIC;
            if isnumeric(val) && ~isempty(val), BIC = val(1); end
        end

        % Print row (use strings, not cells)
        fprintf(fid,'%s & %s & %s & %s & %s & %s & %s \\\\\n', ...
            idxName, mdl, numfmt(omega), numfmt(phi), numfmt(beta), numfmt(d), numfmt(BIC, '%.2f'));
    end

    fprintf(fid,'\\bottomrule\\end{tabular}\\end{table}\n');
    fclose(fid);
end

% -------- helpers (local) --------
function s = aschar(x)
% Return a char row vector (never a cell)
    if isstring(x), s = char(x); 
    elseif ischar(x), s = x;
    elseif iscellstr(x), s = x{1};
    else, s = char(string(x));
    end
end

function v = pickParam(PT, name, def)
% Extract parameter by (case-insensitive) name from table PT.param / PT.est
    v = def;
    if istable(PT) && ismember('param',PT.Properties.VariableNames) && ismember('est',PT.Properties.VariableNames)
        mask = strcmpi(string(PT.param), string(name));
        if any(mask)
            val = PT.est(find(mask,1,'first'));
            if isnumeric(val) && isfinite(val), v = val; end
        end
    end
end

function s = numfmt(x, fmt)
% Format numeric scalar; NaN/empty -> '' (blank). Default fmt '%.3g'
    if nargin<2, fmt = '%.3g'; end
    if ~isnumeric(x) || ~isscalar(x) || ~isfinite(x)
        s = '';
    else
        s = sprintf(fmt, x);
    end
end
