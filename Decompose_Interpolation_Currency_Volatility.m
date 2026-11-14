%% Decompose_Interpolation_Currency_Volatility.m (2026-11-06, fixed)
% HY_native is already a LOG index (R.LogIndex_HY, normalized to 0 at its
% own base period) -- diff() directly, do not re-log. HY (grid) is the
% level index (exp(.)*100) -- diff(log(.)) as before.
clear; clc;
load('RS_Hybrid_Comparison_18.mat','results');

countries = {'ARGFW','CHIFW'};
fprintf('%-8s %10s %10s %8s %8s\n','Series','Native N','NativeVol','GridVol','Ratio');
for i = 1:numel(countries)
    key = countries{i};
    hyN = results.(key).HY_native(:);       % already log-index
    pN  = results.(key).Period_native(:);
    [pN, ord] = sort(pN); hyN = hyN(ord);
    rN = diff(hyN);
    volN = std(rN,'omitnan');

    hyG = results.(key).HY(:);              % level index, base=100
    rG = diff(log(hyG));
    volG = std(rG,'omitnan');

    fprintf('%-8s %10d %10.4f %8.4f %8.2f\n', key, numel(hyN), volN, volG, volN/volG);
end
