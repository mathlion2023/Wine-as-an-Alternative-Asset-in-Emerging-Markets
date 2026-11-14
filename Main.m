%% Main.m
% -------------------------------------------------------------------------
% Orchestrator. Repointed (2026-08-25) at the updated USD master dataset
% (Build_MasterData_USD.m -> MasterData_USD_2013_2024.mat), which replaces
% WineData2025.xlsx as the pipeline's primary input: Jan 2013-Nov 2024
% (143 months, vs. WineData2025.xlsx's Jun 2019-Nov 2024 / 67 months),
% all 20 series in USD and rebased to 100 @ 2013-01.
%
% Original version backed up as Main_ORIGINAL_BACKUP.m. That version read
% WineData2025.xlsx via xlsread and referenced an undefined `assetNames`
% variable in the final corr() display (a pre-existing bug); this version
% takes variable names directly from the master table instead.
% -------------------------------------------------------------------------

clc; clear all;

load('MasterData_USD_2013_2024.mat', 'M');   % USD levels, rebased 100 @ 2013-01

seriesNames = M.Properties.VariableNames(2:end);
Index = M{:, seriesNames};      % T x N level matrix (already base=100 @ 2013-01)
date  = datenum(M.Period);

%% Log-returns
TT = array2table(diff(log(Index)), 'VariableNames', seriesNames);

%% Descriptive statistics (annualized mean/std)
stats = descriptiveStats(TT, 'monthly', true);
disp(stats.table);
writetable(stats.table, 'tab_descr.csv');

%% Monthly return correlation matrix
% LivEx50/LivExInv have shorter history than the rest of the panel (see
% N_eff in stats.table) -- 'Rows','pairwise' keeps their correlations
% instead of propagating NaN into every entry involving those columns.
C = corr(TT{:,:}, 'Rows', 'pairwise');
disp('Monthly return correlation matrix:');
disp(array2table(C, 'VariableNames', seriesNames, 'RowNames', seriesNames));
