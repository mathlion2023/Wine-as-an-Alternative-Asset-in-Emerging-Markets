function [nll, sigma2, eps] = figarch_nll_gauss(y, p, K, burn)
% Gaussian QML for FIGARCH(1,d,1): params p = [omega, phi, beta, d]
% Recursion: (1 - beta L) h_t = omega + [1 - beta L - (1 - phi L)(1 - L)^d] * e_{t-1}^2
% Truncate fractional weights at K lags.

    omega = p(1); phi = p(2); beta = p(3); d = p(4);
    T = numel(y);
    eps = y;                      % innovations (mean-adjusted input)
    % Precompute fractional weights for (1-L)^d
    w = fracdiff_weights(d, K);   % w(1)=1, w(k+1) ~ coeff on L^k
    % Build lambda_k for k>=1:   lambda(L) = 1 - beta L - (1 - phi L)(1-L)^d
    % Expand (1 - phi L)(1-L)^d = w_0 + sum_{k>=1} (w_k - phi * w_{k-1}) L^k
    % Hence, for k>=1: lambda_k = -beta*1_{k=1} - (w_k - phi*w_{k-1})
    lambda = zeros(K,1);  % lambda(1) corresponds to L^1 coefficient
    % k = 1:
    lambda(1) = -beta - (w(2) - phi*w(1));
    % k >= 2:
    for k = 2:K
        lambda(k) = -( w(k+1) - phi*w(k) );
    end
    % Note: constant term works out via recursion as written below.

    % Initialize variance with unconditional-ish level
    h = nan(T,1);
    h0 = var(y,'omitnan');
    h(1:max(2,burn)) = max(1e-10, h0);

    % Compute h_t forward using truncated infinite sum
    % (1 - beta L) h_t = omega + sum_{k=1}^K lambda_k * e_{t-k}^2
    % => h_t = beta * h_{t-1} + omega + sum_{k=1}^{min(K,t-1)} lambda_k * e_{t-k}^2
    for t = max(2,burn+1):T
        acc = 0.0;
        m = min(K, t-1);
        % Accumulate lambda_k * e_{t-k}^2
        for k = 1:m
            ek = eps(t-k);
            if ~isfinite(ek), ek = 0; end
            acc = acc + lambda(k) * (ek*ek);
        end
        ht = beta * h(t-1) + omega + acc;
        h(t) = max(ht, 1e-12); % positivity guard
    end

    sigma2 = h;
    % Gaussian QML log-likelihood
    valid = isfinite(sigma2) & sigma2 > 0;
    % Start from burn+1 to reduce initialization bias
    s = max(burn+1, 2);
    L = 0.5*( log(sigma2(s:end)) + (eps(s:end).^2) ./ sigma2(s:end) );
    nll = sum(L(valid(s:end)));
end

function w = fracdiff_weights(d, K)
% Coefficients for (1 - L)^d = sum_{k=0}^\infty w_k L^k, truncated at K
% Recurrence: w_0 = 1; w_k = w_{k-1} * (k-1 - d)/k
    w = zeros(K+1,1);
    w(1) = 1.0;  % w_0
    for k = 2:K+1
        w(k) = w(k-1) * ((k-2) - d) / (k-1);
    end
end