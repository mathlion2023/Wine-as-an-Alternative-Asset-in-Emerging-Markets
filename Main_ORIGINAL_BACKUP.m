 clc; clear all;

 [data,text]=xlsread("WineData2025.xlsx","Sheet1");
%[data,text]=xlsread("WineData2025.xlsx","RS");
% [data,text]=xlsread("Others.xlsx","Sheet1");
date=datenum(text(2:end,1));

%%
n=size(data);

for i=1:n(2)
X=data(:,i);
b = X(find(~isnan(X),1,'first'));
Index(:,i) = 100 * (X / b);
end

%%


% Assume TT is a timetable with monthly log-returns
%vars = {'Australia','SouthAfrica','Argentina','Chile','LivEx','SPX','MSCIEM','Bonds'};
TT=diff(log(data));

vars = {'SouthAfrica','Australia','Argentina','Chile','SPX','MSCIEM','Bonds','Gold','LivEx'};
stats = descriptiveStats(TT, 'monthly', true);   % annualized mean/std
disp(stats.table);
writetable(stats.table, 'tab_descr.csv');

%%

C = corr(TT);
disp('Monthly return correlation matrix:');
disp(array2table(C, 'vars', assetNames, 'RowNames', assetNames));