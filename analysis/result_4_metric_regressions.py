import sys
sys.path.extend(['../data'])
import json

import numpy as np

import data_processing_funcs as dp
from regression_utils import fit_multivariate_regression, fit_univariate_regression

if __name__ == '__main__':

    K = 8
    LAPSE_THRESHOLD = 2.0

    # Load component coordinates
    coords_BL = np.load(f'../data/logpca_coords_BL_{K}-dims.npy')
    coords_SD = np.load(f'../data/logpca_coords_SD_{K}-dims.npy')

    # Extract first 2 components
    X_BL = coords_BL[:, :2].copy()
    X_SD = coords_SD[:, :2].copy()

    # Apply BL component 2 sign flip (consistent with other analyses)
    X_BL[:, 1] = -X_BL[:, 1]

    # Load and compute behavioral metrics
    rRTs_1, rRTs_2 = dp.get_standard_individual_rRT_data()

    # Compute mean rRT
    mean_rrt_BL = np.array([np.mean(r) for r in rRTs_1])
    mean_rrt_SD = np.array([np.mean(r) for r in rRTs_2])
    
    # Compute rRT standard deviation
    std_rrt_BL = np.array([np.std(r) for r in rRTs_1])
    std_rrt_SD = np.array([np.std(r) for r in rRTs_2])
    
    # Compute lapse rates
    lapse_BL = np.array([np.mean(r < LAPSE_THRESHOLD) for r in rRTs_1])
    lapse_SD = np.array([np.mean(r < LAPSE_THRESHOLD) for r in rRTs_2])


    # Multivariate regression BL
    
    # Stack BL metrics as (n, 3) matrix
    y_BL = np.column_stack([mean_rrt_BL, std_rrt_BL, lapse_BL])
    
    intercept_bl, B_bl, se_matrix_bl, t_stats_bl, p_values_bl, residuals_bl, df_res_bl = fit_multivariate_regression(
        X_BL, y_BL
    )
    
    # Compute R-squared for each response
    ss_tot_bl = np.sum((y_BL - np.mean(y_BL, axis=0))**2, axis=0)
    ss_res_bl = np.sum(residuals_bl**2, axis=0)
    r_sq_bl = 1 - (ss_res_bl / ss_tot_bl)
    
    response_names = ['mean_rRT', 'std_rRT', 'lapse']


    # Multivariate SD
    
    # Stack SD metrics as (n, 3) matrix
    y_SD = np.column_stack([mean_rrt_SD, std_rrt_SD, lapse_SD])
    
    intercept_sd, B_sd, se_matrix_sd, t_stats_sd, p_values_sd, residuals_sd, df_res_sd = fit_multivariate_regression(
        X_SD, y_SD
    )
    
    # Compute R-squared for each response
    ss_tot_sd = np.sum((y_SD - np.mean(y_SD, axis=0))**2, axis=0)
    ss_res_sd = np.sum(residuals_sd**2, axis=0)
    r_sq_sd = 1 - (ss_res_sd / ss_tot_sd)


    # Univariate regression
    
    univariate_results = {}
    
    for cond, X_cond, y_bl_cond, y_sd_cond in [('BL', X_BL, y_BL, None), ('SD', X_SD, None, y_SD)]:
        y_cond = y_bl_cond if cond == 'BL' else y_sd_cond
        
        for resp_j, resp_name in enumerate(response_names):
            y_resp = y_cond[:, resp_j]
            
            for comp_i in range(2):
                key = f'{cond}_{resp_name}_c{comp_i+1}'
                
                result = fit_univariate_regression(X_cond[:, comp_i], y_resp)
                intercept_uni, slope_uni, residuals_uni, r_sq_uni, se_int_uni, se_slope_uni, t_int_uni, t_slope_uni, p_int_uni, p_slope_uni = result
                
                univariate_results[key] = {
                    'intercept': float(intercept_uni),
                    'slope': float(slope_uni),
                    'r_squared': float(r_sq_uni),
                    'se_intercept': float(se_int_uni),
                    'se_slope': float(se_slope_uni),
                    't_intercept': float(t_int_uni),
                    't_slope': float(t_slope_uni),
                    'p_intercept': float(p_int_uni),
                    'p_slope': float(p_slope_uni),
                }

    # BOOTSTRAP ANALYSIS

    from tqdm import tqdm

    bootstrap_results = {
        'multivariate_BL_r2': [],
        'multivariate_SD_r2': [],
        'univariate_BL_mean_rrt_c1_r2': [],
        'univariate_BL_mean_rrt_c2_r2': [],
        'univariate_BL_std_rrt_c1_r2': [],
        'univariate_BL_std_rrt_c2_r2': [],
        'univariate_BL_lapse_c1_r2': [],
        'univariate_BL_lapse_c2_r2': [],
        'univariate_SD_mean_rrt_c1_r2': [],
        'univariate_SD_mean_rrt_c2_r2': [],
        'univariate_SD_std_rrt_c1_r2': [],
        'univariate_SD_std_rrt_c2_r2': [],
        'univariate_SD_lapse_c1_r2': [],
        'univariate_SD_lapse_c2_r2': [],
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

        # Load and process bootstrap RTs
        RTs_dict_bs = dp.json_to_dict(f'../data/bootstrap_samples/RTs/bootstrap_RTs_dict_{bs_idx}.json')
        del RTs_dict_bs['bs_index']
        rRTs_BL_bs, rRTs_SD_bs = dp.get_standard_individual_rRT_data(RTs_dict_bs)

        # Compute behavioral metrics
        mean_rrt_BL_bs = np.array([np.mean(r) for r in rRTs_BL_bs])
        mean_rrt_SD_bs = np.array([np.mean(r) for r in rRTs_SD_bs])
        std_rrt_BL_bs = np.array([np.std(r) for r in rRTs_BL_bs])
        std_rrt_SD_bs = np.array([np.std(r) for r in rRTs_SD_bs])
        lapse_BL_bs = np.array([np.mean(r < LAPSE_THRESHOLD) for r in rRTs_BL_bs])
        lapse_SD_bs = np.array([np.mean(r < LAPSE_THRESHOLD) for r in rRTs_SD_bs])

        # Multivariate regression: BL metrics
        y_BL_bs = np.column_stack([mean_rrt_BL_bs, std_rrt_BL_bs, lapse_BL_bs])
        _, _, _, _, _, resid_bl_bs, _ = fit_multivariate_regression(X_BL_bs, y_BL_bs)
        ss_tot_bl_bs = np.sum((y_BL_bs - np.mean(y_BL_bs, axis=0))**2)
        ss_res_bl_bs = np.sum(resid_bl_bs**2)
        r2_bl_bs = 1 - (ss_res_bl_bs / ss_tot_bl_bs)
        bootstrap_results['multivariate_BL_r2'].append(float(r2_bl_bs))

        # Multivariate regression: SD metrics
        y_SD_bs = np.column_stack([mean_rrt_SD_bs, std_rrt_SD_bs, lapse_SD_bs])
        _, _, _, _, _, resid_sd_bs, _ = fit_multivariate_regression(X_SD_bs, y_SD_bs)
        ss_tot_sd_bs = np.sum((y_SD_bs - np.mean(y_SD_bs, axis=0))**2)
        ss_res_sd_bs = np.sum(resid_sd_bs**2)
        r2_sd_bs = 1 - (ss_res_sd_bs / ss_tot_sd_bs)
        bootstrap_results['multivariate_SD_r2'].append(float(r2_sd_bs))

        # Univariate regressions
        responses_BL = [mean_rrt_BL_bs, std_rrt_BL_bs, lapse_BL_bs]
        responses_SD = [mean_rrt_SD_bs, std_rrt_SD_bs, lapse_SD_bs]
        response_labels = ['mean_rrt', 'std_rrt', 'lapse']

        for resp_idx, resp_label in enumerate(response_labels):
            # BL
            y_resp_bl = responses_BL[resp_idx]
            for comp_i in range(2):
                _, _, _, r2_uni, _, _, _, _, _, _ = fit_univariate_regression(X_BL_bs[:, comp_i], y_resp_bl)
                key = f'univariate_BL_{resp_label}_c{comp_i+1}_r2'
                bootstrap_results[key].append(float(r2_uni))
            
            # SD
            y_resp_sd = responses_SD[resp_idx]
            for comp_i in range(2):
                _, _, _, r2_uni, _, _, _, _, _, _ = fit_univariate_regression(X_SD_bs[:, comp_i], y_resp_sd)
                key = f'univariate_SD_{resp_label}_c{comp_i+1}_r2'
                bootstrap_results[key].append(float(r2_uni))

    # Compute bootstrap means and stds

    bootstrap_stats = {}
    for key, values in bootstrap_results.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        bootstrap_stats[key] = {'mean': float(mean_val), 'std': float(std_val)}
        label = key.replace('_r2', '').replace('_', ' ').upper()

    # SAVE RESULTS
    results = {
        'multivariate_BL': {
            'intercept': intercept_bl.tolist(),
            'B': B_bl.tolist(),
            'r_squared': r_sq_bl.tolist(),
            'se_matrix': se_matrix_bl.tolist(),
            't_stats': t_stats_bl.tolist(),
            'p_values': p_values_bl.tolist(),
            'df_residual': int(df_res_bl),
            'response_names': response_names,
        },
        'multivariate_SD': {
            'intercept': intercept_sd.tolist(),
            'B': B_sd.tolist(),
            'r_squared': r_sq_sd.tolist(),
            'se_matrix': se_matrix_sd.tolist(),
            't_stats': t_stats_sd.tolist(),
            'p_values': p_values_sd.tolist(),
            'df_residual': int(df_res_sd),
            'response_names': response_names,
        },
        'univariate': univariate_results,
        'bootstrap': {
            'n_bootstrap': n_bootstrap,
            'bootstrap_means_and_stds': bootstrap_stats,
        },
    }

    output_file = '../data/behavioral_metrics_component_regression.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
