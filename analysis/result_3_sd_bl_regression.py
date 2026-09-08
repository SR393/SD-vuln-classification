import sys
sys.path.extend(['../data'])
import json
from tqdm import tqdm

import numpy as np
from scipy import stats

import data_processing_funcs as dp
from regression_utils import fit_multivariate_regression_simple, fit_univariate_regression

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    K = 8

    # --- Load BL and SD components ---
    coords_BL = np.load(f'../data/logpca_coords_BL_{K}-dims.npy')
    coords_SD = np.load(f'../data/logpca_coords_SD_{K}-dims.npy')

    # Extract first 2 components
    X_BL = coords_BL[:, :2].copy()
    y_SD = coords_SD[:, :2].copy()

    # Apply BL component 2 sign flip (consistent with other analyses)
    X_BL[:, 1] = -X_BL[:, 1]


    # Multivariate regression

    beta_mult, residuals_mult, r_sq_mult, r_sq_overall_mult = fit_multivariate_regression_simple(X_BL, y_SD)

    intercept_mult = beta_mult[0, :]
    B_mult = beta_mult[1:, :]
    
    n, p = X_BL.shape  # n samples, p predictors (2 components)
    q = y_SD.shape[1]  # q responses (2 components)
    
    # Add intercept column for design matrix
    X_aug_mult = np.column_stack([np.ones(n), X_BL])
    
    # Residual sum of squares and mean squared error per response
    ss_res_mult = np.sum(residuals_mult**2, axis=0)
    df_res_mult = n - (p + 1)  # degrees of freedom (n - number of parameters)
    mse_mult = ss_res_mult / df_res_mult
    
    # Covariance matrix of design
    XtX_inv_mult = np.linalg.inv(X_aug_mult.T @ X_aug_mult)
    
    # Standard errors of all coefficients (intercept + B)
    # SE matrix: (p+1, q) where each element is sqrt(mse * diag_element)
    se_matrix_mult = np.sqrt(np.outer(np.diag(XtX_inv_mult), mse_mult))
    
    # t-statistics and p-values
    t_stats_mult = beta_mult / se_matrix_mult
    p_values_mult = 2 * (1 - stats.t.cdf(np.abs(t_stats_mult), df_res_mult))  # two-tailed

    # Residual analysis (multivariate)
    residual_std_mult = np.std(residuals_mult, axis=0)
    residual_mean_mult = np.mean(residuals_mult, axis=0)

    # Residual covariance matrix
    res_cov_mult = np.cov(residuals_mult.T)

    # Cross-component relationships (unadjusted correlations)
    corr_c1bl_c2sd_mult = np.corrcoef(X_BL[:, 0], y_SD[:, 1])[0, 1]
    corr_c2bl_c1sd_mult = np.corrcoef(X_BL[:, 1], y_SD[:, 0])[0, 1]


    # Univariate regressions
    
    univariate_results = {}
    
    for bl_i in range(2):  # BL components 0, 1 (component 1, 2)
        for sd_j in range(2):  # SD components 0, 1 (component 1, 2)
            key = f'BL_c{bl_i+1}_to_SD_c{sd_j+1}'
            
            result = fit_univariate_regression(X_BL[:, bl_i], y_SD[:, sd_j])
            intercept_uni, slope_uni, residuals_uni, r_sq_uni, se_int_uni, se_slope_uni, t_int_uni, t_slope_uni, p_int_uni, p_slope_uni = result
            
            univariate_results[key] = {
                'intercept': intercept_uni,
                'slope': slope_uni,
                'r_squared': r_sq_uni,
                'se_intercept': se_int_uni,
                'se_slope': se_slope_uni,
                't_intercept': t_int_uni,
                't_slope': t_slope_uni,
                'p_intercept': p_int_uni,
                'p_slope': p_slope_uni,
                'residuals': residuals_uni,
            }

    # Save results
    output = {
        'n_individuals': int(n),
        'multivariate': {
            'intercept': intercept_mult.tolist(),
            'coefficient_matrix_B': B_mult.tolist(),
            'r_squared_per_component': r_sq_mult.tolist(),
            'r_squared_overall': float(r_sq_overall_mult),
            'residual_mean': residual_mean_mult.tolist(),
            'residual_std': residual_std_mult.tolist(),
            'residual_covariance_matrix': res_cov_mult.tolist(),
            'cross_component_correlations': {
                'BL_c1_vs_SD_c2': float(corr_c1bl_c2sd_mult),
                'BL_c2_vs_SD_c1': float(corr_c2bl_c1sd_mult),
            },
            'coefficient_significance': {
                'intercept_coeff': intercept_mult.tolist(),
                'intercept_se': se_matrix_mult[0, :].tolist(),
                'intercept_t': t_stats_mult[0, :].tolist(),
                'intercept_p': p_values_mult[0, :].tolist(),
                'BL_comp1_to_SD': {
                    'coefficients': B_mult[0, :].tolist(),
                    'standard_errors': se_matrix_mult[1, :].tolist(),
                    't_statistics': t_stats_mult[1, :].tolist(),
                    'p_values': p_values_mult[1, :].tolist(),
                },
                'BL_comp2_to_SD': {
                    'coefficients': B_mult[1, :].tolist(),
                    'standard_errors': se_matrix_mult[2, :].tolist(),
                    't_statistics': t_stats_mult[2, :].tolist(),
                    'p_values': p_values_mult[2, :].tolist(),
                },
            },
            'degrees_of_freedom': int(df_res_mult),
        },
        'univariate': {}
    }
    
    # Add univariate results
    for key, result in univariate_results.items():
        output['univariate'][key] = {
            'intercept': float(result['intercept']),
            'slope': float(result['slope']),
            'r_squared': float(result['r_squared']),
            'se_intercept': float(result['se_intercept']),
            'se_slope': float(result['se_slope']),
            't_intercept': float(result['t_intercept']),
            't_slope': float(result['t_slope']),
            'p_intercept': float(result['p_intercept']),
            'p_slope': float(result['p_slope']),
        }

    # bootstrap analysis
    bootstrap_results = {
        'multivariate_c1_r2': [],
        'multivariate_c2_r2': [],
        'univariate_BL_c1_to_SD_c1_r2': [],
        'univariate_BL_c1_to_SD_c2_r2': [],
        'univariate_BL_c2_to_SD_c1_r2': [],
        'univariate_BL_c2_to_SD_c2_r2': [],
    }

    n_bootstrap = 1000

    for bs_idx in tqdm(range(n_bootstrap), desc="Bootstrap samples"):
        # Load bootstrap coordinates
        X_BL_bs = np.load(f'../data/bootstrap_samples/coordinates/logpca_coords_BL_8-dims_{bs_idx}.npy')
        X_SD_bs = np.load(f'../data/bootstrap_samples/coordinates/logpca_coords_SD_8-dims_{bs_idx}.npy')

        # Extract first 2 components
        X_BL_bs = X_BL_bs[:, :2].copy()
        X_SD_bs = X_SD_bs[:, :2].copy()

        # Apply BL component 2 sign flip
        X_BL_bs[:, 1] = -X_BL_bs[:, 1]

        # Multivariate regression: SD ~ BL
        beta_bs, residuals_bs, r_sq_bs, r_sq_overall_bs = fit_multivariate_regression_simple(
            X_BL_bs, X_SD_bs
        )
        bootstrap_results['multivariate_c1_r2'].append(float(r_sq_bs[0]))
        bootstrap_results['multivariate_c2_r2'].append(float(r_sq_bs[1]))

        # Univariate regressions
        for bl_i in range(2):
            for sd_j in range(2):
                _, _, _, r2_uni, _, _, _, _, _, _ = fit_univariate_regression(
                    X_BL_bs[:, bl_i], X_SD_bs[:, sd_j]
                )
                key = f'univariate_BL_c{bl_i+1}_to_SD_c{sd_j+1}_r2'
                bootstrap_results[key].append(float(r2_uni))

    # Compute bootstrap means and stds
    bootstrap_stats = {}
    for key, values in bootstrap_results.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        bootstrap_stats[key] = {'mean': float(mean_val), 'std': float(std_val)}
        label = key.replace('_r2', '').replace('_', ' ').upper()
        print(f"{label}: {mean_val:.6f} ± {std_val:.6f}")

    # Add bootstrap to output
    output['bootstrap'] = {
        'n_bootstrap': n_bootstrap,
        'bootstrap_means_and_stds': bootstrap_stats
    }

    out_path = '../data/sd_bl_component_regression.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
