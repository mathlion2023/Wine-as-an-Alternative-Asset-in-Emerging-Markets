T=readtable('Data Australia.xlsx','Sheet','Sheet1');
T.ymDate=datetime(T.ym,'InputFormat','yyyy-MM');
ctys = {'Argentina','Australia','Chile'};
for i=1:numel(ctys)
  cc = ctys{i};
  Tc=T(strcmp(T.wine_country,cc),:);
  fprintf('%s: min=%s max=%s n=%d\n', cc, datestr(min(Tc.ymDate)), datestr(max(Tc.ymDate)), height(Tc));
end
