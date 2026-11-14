load('ResidualSTD_4country.mat');
C = corr(stdResid, 'rows','pairwise');
names = string(names);
livIdx = find(names=="LivEx100_USD");
wineIdx = find(ismember(names, ["SA_wine_USD","AUS_wine_USD","ARG_wine_USD","CHL_wine_USD"]));
fprintf('Unconditional correlation with Liv-ex100:\n');
for i = wineIdx(:)'
    fprintf('  %s: %.3f\n', names(i), C(i,livIdx));
end
offdiag = C(~eye(size(C)));
fprintf('System-wide average |unconditional correlation| (off-diag): %.3f\n', mean(abs(offdiag)));
fprintf('System-wide average unconditional correlation (off-diag, signed): %.3f\n', mean(offdiag));
