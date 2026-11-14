function gen_modelsel_table(csvFname, keys_, labels_, outFname, repeatIndex)
% Generates LaTeX body rows for the 4-candidate model comparison table.
% keys_/labels_: parallel cell arrays, raw series key -> display name, in desired row order.
if nargin<5, repeatIndex = true; end
T = readtable(csvFname);
fid = fopen(outFname,'w');
for ii = 1:numel(keys_)
    key = keys_{ii};
    disp_ = labels_{ii};
    sub = T(strcmp(T.Index, key), :);
    order = {'GARCH','GJR','EGARCH','FIGARCH'};
    bics = nan(1,4); logLs = nan(1,4); aics = nan(1,4);
    valid = true(1,4);
    for k=1:4
        r = sub(strcmpi(sub.Model, order{k}),:);
        if isempty(r), valid(k)=false; continue; end
        if ~isfinite(r.BIC) || abs(r.BIC) > 1e6
            valid(k) = false;
        else
            bics(k) = r.BIC; logLs(k)=r.logL; aics(k)=r.AIC;
        end
    end
    [~, bestK] = min(bics);
    for k = 1:4
        r = sub(strcmpi(sub.Model, order{k}),:);
        nm = order{k};
        if k==bestK, nm = ['\textbf{' nm '}']; end
        if isempty(r)
            logLstr='--'; aicstr='--'; bicstr='--';
        elseif ~valid(k)
            logLstr='--$^{\ddagger}$'; aicstr='--'; bicstr='--';
        else
            logLstr = sprintf('%.3f', r.logL);
            aicstr = sprintf('%.3f', r.AIC);
            bicstr = sprintf('%.3f', r.BIC);
        end
        dcell = '';
        if strcmpi(order{k},'FIGARCH') && ~isempty(r) && isfinite(r.d)
            dcell = sprintf('%.3f (%.2f)', r.d, r.tstat_d);
        end
        idxCell = disp_;
        if ~repeatIndex && k>1, idxCell = ''; end
        fprintf(fid, '%s & %s & %s & %s & %s & %s \\\\\n', idxCell, nm, logLstr, aicstr, bicstr, dcell);
    end
    if ii < numel(keys_)
        fprintf(fid, '\\addlinespace\n');
    end
end
fclose(fid);
end
