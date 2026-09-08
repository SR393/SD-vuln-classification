import numpy as np
from scipy import stats


def fit_multivariate_regression(X, y):

    n, p = X.shape
    q = y.shape[1] if len(y.shape) > 1 else 1
    
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    
    # Add intercept column
    X_aug = np.column_stack([np.ones(n), X])
    
    df_residual = n - (p + 1)  # residual degrees of freedom
    
    # Solve all q systems at once: OLS: beta = (X'X)^-1 X'Y
    beta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
    # beta shape: (p+1, q)
    
    # Compute residuals (n, q)
    residuals = y - X_aug @ beta
    
    # Compute MSE per response: (q,)
    mse = np.sum(residuals**2, axis=0) / df_residual
    
    # Compute (X'X)^-1 once (same for all responses)
    XtX_inv = np.linalg.pinv(X_aug.T @ X_aug)
    se_diag = np.sqrt(np.diag(XtX_inv))  # (p+1,)
    
    # Standard errors: broadcast se_diag with mse
    # se_matrix = se_diag[:, None] * sqrt(mse[None, :])
    se_matrix = se_diag[:, None] * np.sqrt(mse[None, :])  # (p+1, q)
    
    # t-statistics: coefficient / se (element-wise)
    t_stats = beta / se_matrix  # (p+1, q)
    
    # p-values: vectorized (p+1, q)
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df_residual))
    
    intercept = beta[0, :]
    B = beta[1:, :]
    
    return intercept, B, se_matrix, t_stats, p_values, residuals, df_residual


def fit_multivariate_regression_simple(X, y):

    n, p = X.shape
    
    # Add intercept column
    X_aug = np.column_stack([np.ones(n), X])
    
    # Solve all q systems at once
    beta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
    
    # Residuals
    residuals = y - X_aug @ beta
    
    # R-squared per response
    ss_res = np.sum(residuals**2, axis=0)
    ss_tot = np.sum((y - np.mean(y, axis=0))**2, axis=0)
    r_squared = 1 - ss_res / ss_tot
    
    # Overall R-squared (fraction of total variance explained)
    ss_res_overall = np.sum(residuals**2)
    ss_tot_overall = np.sum((y - np.mean(y))**2)
    r_squared_overall = 1 - ss_res_overall / ss_tot_overall
    
    return beta, residuals, r_squared, r_squared_overall



def fit_univariate_regression(X, y):

    n = len(X)
    X_aug = np.column_stack([np.ones(n), X])
    
    beta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
    intercept, slope = beta
    
    residuals = y - X_aug @ beta
    
    # R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - ss_res / ss_tot
    
    # Standard errors
    mse = ss_res / (n - 2)  # 2 parameters (intercept + slope)
    XtX_inv = np.linalg.inv(X_aug.T @ X_aug)
    se_vector = np.sqrt(np.diag(XtX_inv) * mse)
    se_intercept, se_slope = se_vector
    
    # t-statistics and p-values
    t_intercept = intercept / se_intercept
    t_slope = slope / se_slope
    p_intercept = 2 * (1 - stats.t.cdf(np.abs(t_intercept), n - 2))
    p_slope = 2 * (1 - stats.t.cdf(np.abs(t_slope), n - 2))
    
    return intercept, slope, residuals, r_squared, se_intercept, se_slope, t_intercept, t_slope, p_intercept, p_slope
