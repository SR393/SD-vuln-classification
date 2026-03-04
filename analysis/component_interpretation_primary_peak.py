import sys
sys.path.extend(['../data'])
import os
import multiprocessing as mproc
from functools import partial
from tqdm import tqdm

import numpy as np
from scipy.stats import gaussian_kde
from KDEpy import FFTKDE

import data_processing_funcs as dp
from log_pca_analysis_funcs import (geodesic_quantiles_from_component, get_empirical_peaks_all, 
                                    pdf_from_quantiles_arr)

def get_all_correlations(index, alphas, x_grid, emp_peak_locs_BL, emp_peak_locs_SD):

    max_K = 8

    if not index is None:

        log_Qbar_BL = np.load(f'../data/bootstrap_samples/q_means/logpca_frechet_mean_BL_{max_K}-dims_{index}.npy')
        components_BL = np.load(f'../data/bootstrap_samples/components/logpca_components_BL_{max_K}-dims_{index}.npy')
        beta_BL = np.load(f'../data/bootstrap_samples/coordinates/logpca_coords_BL_{max_K}-dims_{index}.npy')

        log_Qbar_SD = np.load(f'../data/bootstrap_samples/q_means/logpca_frechet_mean_SD_{max_K}-dims_{index}.npy')
        components_SD = np.load(f'../data/bootstrap_samples/components/logpca_components_SD_{max_K}-dims_{index}.npy')
        beta_SD = np.load(f'../data/bootstrap_samples/coordinates/logpca_coords_SD_{max_K}-dims_{index}.npy')
    
    else:

        log_Qbar_BL = np.load(f'../data/logpca_frechet_mean_BL_{max_K}-dims.npy')
        components_BL = np.load(f'../data/logpca_components_BL_{max_K}-dims.npy')
        beta_BL = np.load(f'../data/logpca_coords_BL_{max_K}-dims.npy')

        log_Qbar_SD = np.load(f'../data/logpca_frechet_mean_SD_{max_K}-dims.npy')
        components_SD = np.load(f'../data/logpca_components_SD_{max_K}-dims.npy')
        beta_SD = np.load(f'../data/logpca_coords_SD_{max_K}-dims.npy')

    comp_1_projection_quantiles_BL = np.array(geodesic_quantiles_from_component(log_Qbar_BL, components_BL[0], t_vals=beta_BL[:, 0])[1])
    comp_1_projection_quantiles_SD = np.array(geodesic_quantiles_from_component(log_Qbar_SD, components_SD[0], t_vals=beta_SD[:, 0])[1])
    comp_2_projection_quantiles_BL = np.array(geodesic_quantiles_from_component(log_Qbar_BL, components_BL[1], t_vals=beta_BL[:, 1])[1])
    comp_2_projection_quantiles_SD = np.array(geodesic_quantiles_from_component(log_Qbar_SD, components_SD[1], t_vals=beta_SD[:, 1])[1])

    ## Comp 1 BL ##
                                                 
    comp_1_projection_densities_BL = pdf_from_quantiles_arr(comp_1_projection_quantiles_BL, alphas, x_grid=x_grid)[1]
    primary_peak_locs_comp_1_BL = get_empirical_peaks_all(x_grid, comp_1_projection_densities_BL)[0]
    
    comp_1_BL_correlation = np.corrcoef(primary_peak_locs_comp_1_BL, emp_peak_locs_BL)[0, 1]

    ## Comp 1 SD ##
                                                        
    comp_1_projection_densities_SD = pdf_from_quantiles_arr(comp_1_projection_quantiles_SD, alphas, x_grid=x_grid)[1]
    primary_peak_locs_comp_1_SD = get_empirical_peaks_all(x_grid, comp_1_projection_densities_SD)[0]

    comp_1_SD_correlation = np.corrcoef(primary_peak_locs_comp_1_SD, emp_peak_locs_SD)[0, 1]

    ## Comp 2 BL ##
    comp_2_projection_densities_BL = pdf_from_quantiles_arr(comp_2_projection_quantiles_BL, alphas, x_grid=x_grid)[1]
    primary_peak_locs_comp_2_BL = get_empirical_peaks_all(x_grid, comp_2_projection_densities_BL)[0]

    comp_2_BL_correlation = np.corrcoef(primary_peak_locs_comp_2_BL, emp_peak_locs_BL)[0, 1]

    ## Comp 2 SD ##
    comp_2_projection_densities_SD = pdf_from_quantiles_arr(comp_2_projection_quantiles_SD, alphas, x_grid=x_grid)[1]
    primary_peak_locs_comp_2_SD = get_empirical_peaks_all(x_grid, comp_2_projection_densities_SD)[0]

    comp_2_SD_correlation = np.corrcoef(primary_peak_locs_comp_2_SD, emp_peak_locs_SD)[0, 1]

    return comp_1_BL_correlation, comp_1_SD_correlation, comp_2_BL_correlation, comp_2_SD_correlation

if __name__ == "__main__":

    x_grid = np.linspace(-1.0, 6.5, 2500)

    rRTs_1, rRTs_2 = dp.get_standard_individual_rRT_data()
    # emp_kdes_BL = np.array([gaussian_kde(rRTs)(x_grid) for rRTs in rRTs_1])
    # emp_kdes_SD = np.array([gaussian_kde(rRTs)(x_grid) for rRTs in rRTs_2])
    emp_kdes_BL = np.array([FFTKDE(bw='silverman').fit(rRTs).evaluate(x_grid) for rRTs in rRTs_1])
    emp_kdes_SD = np.array([FFTKDE(bw='silverman').fit(rRTs).evaluate(x_grid) for rRTs in rRTs_2])

    emp_peak_locs_BL = get_empirical_peaks_all(x_grid, emp_kdes_BL, rRTs_1)[0]
    emp_peak_locs_SD = get_empirical_peaks_all(x_grid, emp_kdes_SD, rRTs_2)[0]

    max_K = 8

    eps = 1e-3
    M = 200
    alphas = np.linspace(eps, 1 - eps, M)

    if not os.path.exists('../data/logpca_primary_peak_location_correlations.json'):
        
        n_repeats = 1000

        corrs_func = partial(get_all_correlations, emp_peak_locs_BL = emp_peak_locs_BL, emp_peak_locs_SD = emp_peak_locs_SD, 
                            x_grid = x_grid, alphas = alphas)

        output = []
        for i in tqdm(range(n_repeats), total = n_repeats):
            output.append(corrs_func(i))

        output = np.array(output)
        comp_1_BL_correlations = output[:, 0]
        comp_1_SD_correlations = output[:, 1]
        comp_2_BL_correlations = output[:, 2]
        comp_2_SD_correlations = output[:, 3]

        comp_1_BL_std = np.std(comp_1_BL_correlations)
        comp_1_SD_std = np.std(comp_1_SD_correlations)
        comp_2_BL_std = np.std(comp_2_BL_correlations)
        comp_2_SD_std = np.std(comp_2_SD_correlations)

        correlations_dict = {
            'comp_1_BL': np.array([np.mean(comp_1_BL_correlations), comp_1_BL_std]).tolist(),
            'comp_1_SD': np.array([np.mean(comp_1_SD_correlations), comp_1_SD_std]).tolist(),
            'comp_2_BL': np.array([np.mean(comp_2_BL_correlations), comp_2_BL_std]).tolist(),
            'comp_2_SD': np.array([np.mean(comp_2_SD_correlations), comp_2_SD_std]).tolist()
        }

        dp.dict_to_json(correlations_dict, f'../data/logpca_primary_peak_location_correlation_stats.json')
    else:
        correlations_dict = dp.json_to_dict(f'../data/logpca_primary_peak_location_correlation_stats.json')
        comp_1_BL_std = correlations_dict['comp_1_BL'][1]
        comp_1_SD_std = correlations_dict['comp_1_SD'][1]
        comp_2_BL_std = correlations_dict['comp_2_BL'][1]
        comp_2_SD_std = correlations_dict['comp_2_SD'][1]
    
    true_correlations = get_all_correlations(None, emp_peak_locs_BL = emp_peak_locs_BL, emp_peak_locs_SD = emp_peak_locs_SD, 
                                             x_grid = x_grid, alphas = alphas)
    
    print(f"Comp 1 BL locs corr: {true_correlations[0]:.3f} +- {comp_1_BL_std:.3f}")
    print(f"Comp 1 SD locs corr: {true_correlations[1]:.3f} +- {comp_1_SD_std:.3f}")
    print(f"Comp 2 BL locs corr: {true_correlations[2]:.3f} +- {comp_2_BL_std:.3f}")
    print(f"Comp 2 SD locs corr: {true_correlations[3]:.3f} +- {comp_2_SD_std:.3f}")

    import pdb; pdb.set_trace()