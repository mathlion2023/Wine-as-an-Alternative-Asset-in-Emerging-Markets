function write_latex_model_table_all(OUT, names, fname)
% WRITE_LATEX_MODEL_TABLE  Export selected model params + t-stats + logL/AIC/BIC to LaTeX.
%   write_latex_model_table(OUT, names, 'tab_selected_models.tex')
%
% Expected OUT (from fit_auto_vol_models):
%   OUT.choice{i}  : model name e.g. 'FIGARCH', 'GARCH', ...
%   OUT.params{i}  : table with columns {'param','est','se'} (se may be NaN)
%   OUT.stats{i}   : table with {'logL','AIC','BIC'}
%
% Output columns:
%   Index | Model | omega | phi | beta | d | logL | AIC | BIC
% Each param cell is printed as "est (tstat)"; if se missing, prints just "est".
%
% Notes:
% - For non-FIGARCH models that don’t have 'omega','phi','beta','d', the cell is left blank.
% - Requires LaTeX packages: booktabs (for \toprule etc.) in your .tex preamble.

    if iscell(names), names = string(names); end
    fid = fopen(fname,'w');  assert(fid>0, 'Cannot open "%s" for writing.', fname);

    fprintf(fid,'\\begin{table}[t]\\centering\\caption{Selected volatility models with parameter estimates (t\\,stats in parentheses).}\\label{tab:volsel}\n');
    fprintf(fid,'\\begin{tabular}{l l l l l l r r r}\\toprule\n');
    fprintf(fid,'Index & Model & $\\omega$ & $\\phi$ & $\\beta$ & $d$ & logL & AIC & BIC \\\\\\midrule\n');

    N = numel(names);
    for i = 1:N
        % Skip rows with no selection
        if i>numel(OUT.choice) || isempty(OUT.choice{i})
            continue;
        end

        idxName = aschar(names(i));
        mdl     = aschar(OUT.choice{i});

        % Defaults
        omegaS = ''; phiS = ''; betaS = ''; dS = '';
        logL_ = NaN; AIC_ = NaN; BIC_ = NaN;

        % Parameters (if available)
        if i<=numel(OUT.params) && ~isempty(OUT.params{i})
            PT = OUT.params{i};   % table with param/est/se (case-insensitive)
            omegaS = fmtParam(PT, 'omega');
            phiS   = fmtParam(PT, 'phi');
            betaS  = fmtParam(PT, 'beta');
            dS     = fmtParam(PT, 'd');      % will stay '' if not present
        end

        % Stats (if available)
        if i<=numel(OUT.stats) && ~isempty(OUT.stats{i}) && istable(OUT.stats{i})
            logL_ = getScalarField(OUT.stats{i}, 'logL');
            AIC_  = getScalarField(OUT.stats{i}, 'AIC');
            BIC_  = getScalarField(OUT.stats{i}, 'BIC');
        end

        % Print one row (all as strings except last three numeric formatted here)
        fprintf(fid,'%s & %s & %s & %s & %s & %s & %s & %s & %s \\\\\n', ...
            idxName, mdl, omegaS, phiS, betaS, dS, ...
            numfmt(logL_, '%.3f'), numfmt(AIC_, '%.3f'), numfmt(BIC_, '%.3f'));
    end

    fprintf(fid,'\\bottomrule\\end{tabular}\n');
    fprintf(fid,'\\begin{flushleft}\\footnotesize ');
    fprintf(fid,'\\emph{Notes:} Entries are QML estimates; t\\,statistics in parentheses use Hessian-based s.e. where available. ');
    fprintf(fid,'Cells are left blank when a parameter is not part of the selected model. ');
    fprintf(fid,'logL, AIC, and BIC refer to the selected model per index.');
    fprintf(fid,'\\end{flushleft}\\end{table}\n');
    fclose(fid);
end

% ---------- helpers ----------

function s = fmtParam(PT, pname)
% Return "est (t)" string for parameter pname from table PT(param,est,se).
% If not found or est invalid -> ''.
% If se missing/NaN -> just "est".
    s = '';
    if ~istable(PT) || ~all(ismember({'param','est','se'}, PT.Properties.VariableNames))
        return;
    end
    mask = strcmpi(string(PT.param), string(pname));
    if ~any(mask), return; end
    est = PT.est(find(mask,1,'first'));
    if ~isnumeric(est) || ~isscalar(est) || ~isfinite(est)
        return;
    end
    % try t-stat
    se  = PT.se(find(mask,1,'first'));
    if isnumeric(se) && isscalar(se) && isfinite(se) && se>0
        t = est / se;
        s = sprintf('%.3g (%.2f)', est, t);
    else
        s = sprintf('%.3g', est);
    end
end

function v = getScalarField(T, name)
% Pull a numeric scalar from table T column 'name'; otherwise NaN.
    v = NaN;
    if istable(T) && ismember(name, T.Properties.VariableNames)
        val = T.(name);
        if isnumeric(val) && ~isempty(val)
            v = val(1);
        end
    end
end

function s = aschar(x)
% Convert to a plain char row vector (no cells).
    if isstring(x), s = char(x);
    elseif ischar(x), s = x;
    elseif iscellstr(x), s = x{1};
    else, s = char(string(x));
    end
end

function s = numfmt(x, fmt)
% Format numeric scalar; NaN -> '' (blank).
    if nargin<2, fmt = '%.3g'; end
    if ~isnumeric(x) || ~isscalar(x) || ~isfinite(x)
        s = '';
    else
        s = sprintf(fmt, x);
    end
end
