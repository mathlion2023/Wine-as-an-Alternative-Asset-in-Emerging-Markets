%% Align + compute basic comparisons
% Rebase both to the same base month (first common month)

clc; clear all;

load('Projectdata.mat');
load('Repeatsales.mat');
commonDates=RS(:,1);

dataRS=RS(:,2:end);
dataHY=data(:,1:4);

%%
id=2;

RS=dataRS(:,id);
HY=dataHY(:,id);



% Levels and log-returns
logRS = log(RS); logHY = log(HY);
rRS   = diff(logRS); rHY = diff(logHY);

% 1) Path metrics
rmsad_levels = sqrt(mean((logRS - logHY).^2));      % RMS dev of log levels
maxgap_levels= max(abs(logRS - logHY));             % max abs gap in log
corr_returns = corr(rRS, rHY, 'Rows','complete');   % corr of monthly returns

% 2) Return diagnostics
stats = @(x) [mean(x)*12, std(x)*sqrt(12), (mean(x)*12)/(std(x)*sqrt(12)), skewness(x), kurtosis(x)-3];
S_RS = stats(rRS);  % [annMean, annVol, Sharpe0, Skew, ExKurt]
S_HY = stats(rHY);

fprintf('RMSAD(log levels) = %.4f | Max |Δlog| = %.4f | Corr(ret) = %.2f\n', rmsad_levels, maxgap_levels, corr_returns);
fprintf('RS  annMean=%.2f%% annVol=%.2f%% Sharpe=%.2f Sk=%.2f ExK=%.2f\n', S_RS(1)*100, S_RS(2)*100, S_RS(3), S_RS(4), S_RS(5));
fprintf('HY  annMean=%.2f%% annVol=%.2f%% Sharpe=%.2f Sk=%.2f ExK=%.2f\n', S_HY(1)*100, S_HY(2)*100, S_HY(3), S_HY(4), S_HY(5));
Res=[rmsad_levels maxgap_levels corr_returns S_HY ];
%%
% 3) Cointegration (Engle–Granger style via ADF on residual)
% logRS_t = a + b logHY_t + u_t ; ADF on u_t
X = [ones(size(logHY)) logHY];
ab = X \ logRS; u = logRS - X*ab;
[h_adf, p_adf] = adftest(u,'Model','ARD','Lags',2);  % H0: unit root
fprintf('EG cointegration ADF: reject unit root? h=%d (p=%.3f)\n', h_adf, p_adf);
%%
% 4) Diebold–Mariano on 1-step forecasts of returns (simple AR(1) benchmark)
% Fit AR(1) on each and compare absolute forecast errors
T = numel(rRS);
y1 = rRS(2:end); X1 = [ones(T-1,1), rRS(1:end-1)];
b1 = X1 \ y1; e1 = y1 - X1*b1;

y2 = rHY(2:end); X2 = [ones(T-1,1), rHY(1:end-1)];
b2 = X2 \ y2; e2 = y2 - X2*b2;

loss1 = abs(e1); loss2 = abs(e2);
d = loss1 - loss2;               % positive => RS worse than HY
dm_stat = mean(d) / (std(d)/sqrt(numel(d)));
p_dm = 2*(1 - normcdf(abs(dm_stat)));
fprintf('DM test (abs 1-step errors): stat=%.2f, p=%.3f (positive => HY better)\n', dm_stat, p_dm);
%%
% 5) Plots: levels, gaps, rolling corr
figure('Color','w'); 
subplot(3,1,1);
plot(commonDates, RS, '-','LineWidth',1.2); hold on;
plot(commonDates, HY, '--','LineWidth',1.2); grid on;
legend('Repeat-Sales','Hybrid','Location','best'); ylabel('Index (base=100)'); title('RS vs Hybrid: Levels');
datetick
axis tight
subplot(3,1,2);
plot(commonDates, 100*(logRS - logHY), 'LineWidth',1.1); grid on;
ylabel('Δlog(levels) × 100'); title('Gap (log RS - log HY)');
datetick
axis tight
subplot(3,1,3);
w = 36;  % rolling window
rollcorr = nan(numel(rRS),1);
for t=w:numel(rRS)
    rollcorr(t) = corr(rRS(t-w+1:t), rHY(t-w+1:t), 'Rows','complete');
end
plot(commonDates(2:end), rollcorr, 'LineWidth',1.1); grid on;
ylabel('Rolling 36m corr'); xlabel('Date');
datetick
axis tight