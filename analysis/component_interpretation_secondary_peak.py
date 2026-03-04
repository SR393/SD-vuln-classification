## TODO: Add repeat sampling for uncertainty estimation ##

import sys
sys.path.extend(['../data'])
import os
import multiprocessing as mproc
from functools import partial
from tqdm import tqdm

import numpy as np
from scipy.stats import gaussian_kde
from KDEpy import FFTKDE
import fastkde

import data_processing_funcs as dp
from log_pca_analysis_funcs import (geodesic_quantiles_from_component, pdf_from_quantiles_arr, 
                                    get_empirical_peaks_all, reconstruct_quantiles)

def get_metrics(index, alphas, x_grid, x1_ind, emp_secondary_peak_locs, emp_left_tail_mass):
    
    K = 8

    if not index is None:
        log_Qbar = np.load(f'../data/bootstrap_samples/q_means/logpca_frechet_mean_SD_{K}-dims_{index}.npy')
        components = np.load(f'../data/bootstrap_samples/components/logpca_components_SD_{K}-dims_{index}.npy')
        beta = np.load(f'../data/bootstrap_samples/coordinates/logpca_coords_SD_{K}-dims_{index}.npy')
    else:
        log_Qbar = np.load(f'../data/logpca_frechet_mean_SD_{K}-dims.npy')
        components = np.load(f'../data/logpca_components_SD_{K}-dims.npy')
        beta = np.load(f'../data/logpca_coords_SD_{K}-dims.npy')
    
    comp_1_projection_quantiles = np.array(geodesic_quantiles_from_component(log_Qbar, components[0], t_vals=beta[:, 0])[1])
    comp_2_projection_quantiles = np.array(geodesic_quantiles_from_component(log_Qbar, components[1], t_vals=beta[:, 1])[1])
    both_components_projection_quantiles = reconstruct_quantiles(log_Qbar, components[:2], beta[:, :2])

    ## Comp 1 ##
                                 
    comp_1_projection_densities = pdf_from_quantiles_arr(comp_1_projection_quantiles, alphas, x_grid=x_grid)[1]
    secondary_peak_locs_comp_1 = get_empirical_peaks_all(x_grid, comp_1_projection_densities)[1]

    comp_1_correctly_has_sp = (~np.isnan(emp_secondary_peak_locs))*(~np.isnan(secondary_peak_locs_comp_1))
    comp_1_proportion_correct = np.sum(comp_1_correctly_has_sp) / len(comp_1_correctly_has_sp)
    comp_1_correct_sp_inds = np.nonzero(comp_1_correctly_has_sp)[0]
    comp_1_left_tail_mass = np.array([np.trapz(dens[:x1_ind], x_grid[:x1_ind]) for dens in comp_1_projection_densities])
    comp_1_sp_loc_correlation = np.corrcoef(secondary_peak_locs_comp_1[comp_1_correct_sp_inds], emp_secondary_peak_locs[comp_1_correct_sp_inds])[0, 1]
    comp_1_left_tail_mass_correlation = np.corrcoef(comp_1_left_tail_mass[comp_1_correct_sp_inds], emp_left_tail_mass[comp_1_correct_sp_inds])[0, 1]
    comp_1_proportion_correct = comp_1_proportion_correct

    ## Comp 2 SD ##                                          
    comp_2_projection_densities = pdf_from_quantiles_arr(comp_2_projection_quantiles, alphas, x_grid=x_grid)[1] 
    secondary_peak_locs_comp_2 = get_empirical_peaks_all(x_grid, comp_2_projection_densities)[1]

    comp_2_correctly_has_sp = (~np.isnan(emp_secondary_peak_locs))*(~np.isnan(secondary_peak_locs_comp_2))
    comp_2_proportion_correct = np.sum(comp_2_correctly_has_sp) / len(comp_2_correctly_has_sp)
    comp_2_correct_sp_inds = np.nonzero(comp_2_correctly_has_sp)[0]
    comp_2_left_tail_mass = np.array([np.trapz(dens[:x1_ind], x_grid[:x1_ind]) for dens in comp_2_projection_densities])
    comp_2_sp_loc_correlation = np.corrcoef(secondary_peak_locs_comp_2[comp_2_correct_sp_inds], emp_secondary_peak_locs[comp_2_correct_sp_inds])[0, 1]
    comp_2_left_tail_mass_correlation = np.corrcoef(comp_2_left_tail_mass[comp_2_correct_sp_inds], emp_left_tail_mass[comp_2_correct_sp_inds])[0, 1]
    comp_2_proportion_correct = comp_2_proportion_correct

    ## Both Comps SD ##                                                   
    both_projection_densities = pdf_from_quantiles_arr(both_components_projection_quantiles, alphas, x_grid=x_grid)[1]
    secondary_peak_locs_both = get_empirical_peaks_all(x_grid, both_projection_densities)[1]

    both_correctly_has_sp = (~np.isnan(emp_secondary_peak_locs))*(~np.isnan(secondary_peak_locs_both))
    both_proportion_correct = np.sum(both_correctly_has_sp) / len(both_correctly_has_sp)
    both_correct_sp_inds = np.nonzero(both_correctly_has_sp)[0]
    both_left_tail_mass = np.array([np.trapz(dens[:x1_ind], x_grid[:x1_ind]) for dens in both_projection_densities])
    both_sp_loc_correlation = np.corrcoef(secondary_peak_locs_both[both_correct_sp_inds], emp_secondary_peak_locs[both_correct_sp_inds])[0, 1]
    both_left_tail_mass_correlation = np.corrcoef(both_left_tail_mass[both_correct_sp_inds], emp_left_tail_mass[both_correct_sp_inds])[0, 1]
    both_proportion_correct = both_proportion_correct

    return comp_1_sp_loc_correlation, comp_1_left_tail_mass_correlation, comp_1_proportion_correct, \
           comp_2_sp_loc_correlation, comp_2_left_tail_mass_correlation, comp_2_proportion_correct, \
           both_sp_loc_correlation, both_left_tail_mass_correlation, both_proportion_correct

if __name__ == "__main__":

    x_grid = np.linspace(-1.0, 6.5, 1000)
    x1_ind = np.where(x_grid >= 1.0)[0][0]
    dx = x_grid[1] - x_grid[0]
    if not os.path.exists('../data/logpca_empirical_sp_locs_SD.npy'):
        _, rRTs_2 = dp.get_standard_individual_rRT_data()
        emp_kdes_SD = np.array([FFTKDE(bw='silverman').fit(rRTs).evaluate(x_grid) for rRTs in rRTs_2])
        emp_secondary_peak_locs_SD = get_empirical_peaks_all(x_grid, emp_kdes_SD, rRTs_2)[1]
        emp_left_tail_mass_SD = np.array([np.trapz(kde[:x1_ind], x_grid[:x1_ind]) for kde in emp_kdes_SD])
        np.save('../data/logpca_empirical_sp_locs_SD.npy', emp_secondary_peak_locs_SD)
        np.save('../data/logpca_empirical_left_tail_mass_SD.npy', emp_left_tail_mass_SD)
    else:
        emp_secondary_peak_locs_SD = np.load('../data/logpca_empirical_sp_locs_SD.npy')
        emp_left_tail_mass_SD = np.load('../data/logpca_empirical_left_tail_mass_SD.npy')

    max_K = 8
    eps = 1e-3
    M = 200
    alphas = np.linspace(eps, 1 - eps, M)

    if not os.path.exists('../data/logpca_secondary_peak_correlations.json'):

        n_repeats = 1000

        mets_func = partial(get_metrics, emp_secondary_peak_locs = emp_secondary_peak_locs_SD, 
                            emp_left_tail_mass = emp_left_tail_mass_SD,
                            x_grid = x_grid, alphas = alphas, x1_ind = x1_ind)
        output = []
        for repeat in tqdm(range(n_repeats)):
            output.append(mets_func(repeat))

        output = np.array(output)

        comp_1_SD_sp_loc_correlations = output[:, 0]
        comp_1_SD_left_tail_mass_correlations = output[:, 1]
        comp_1_SD_proportion_correct = output[:, 2]

        comp_2_SD_sp_loc_correlations = output[:, 3]
        comp_2_SD_left_tail_mass_correlations = output[:, 4]
        comp_2_SD_proportion_correct = output[:, 5]

        both_SD_sp_loc_correlations = output[:, 6]
        both_SD_left_tail_mass_correlations = output[:, 7]
        both_SD_proportion_correct = output[:, 8]

        comp_1_sp_loc_std = np.nanstd(comp_1_SD_sp_loc_correlations)
        comp_1_left_tail_mass_std = np.nanstd(comp_1_SD_left_tail_mass_correlations)
        comp_1_proportion_correct_std = np.nanstd(comp_1_SD_proportion_correct)

        comp_2_sp_loc_std = np.nanstd(comp_2_SD_sp_loc_correlations)
        comp_2_left_tail_mass_std = np.nanstd(comp_2_SD_left_tail_mass_correlations)
        comp_2_proportion_correct_std = np.nanstd(comp_2_SD_proportion_correct)

        both_sp_loc_std = np.nanstd(both_SD_sp_loc_correlations)
        both_left_tail_mass_std = np.nanstd(both_SD_left_tail_mass_correlations)
        both_proportion_correct_std = np.nanstd(both_SD_proportion_correct)

        correlations_dict = {
            'comp_1_SP_locs': np.array([np.nanmean(comp_1_SD_sp_loc_correlations), comp_1_sp_loc_std]).tolist(),
            'comp_1_left_tail_mass': np.array([np.nanmean(comp_1_SD_left_tail_mass_correlations), comp_1_left_tail_mass_std]).tolist(),
            'comp_1_proportion_correct': np.array([np.nanmean(comp_1_SD_proportion_correct), comp_1_proportion_correct_std]).tolist(),
            'comp_2_SP_locs': np.array([np.nanmean(comp_2_SD_sp_loc_correlations), comp_2_sp_loc_std]).tolist(),
            'comp_2_left_tail_mass': np.array([np.nanmean(comp_2_SD_left_tail_mass_correlations), comp_2_left_tail_mass_std]).tolist(),
            'comp_2_proportion_correct': np.array([np.nanmean(comp_2_SD_proportion_correct), comp_2_proportion_correct_std]).tolist(),
            'both_SP_locs': np.array([np.nanmean(both_SD_sp_loc_correlations), both_sp_loc_std]).tolist(),
            'both_left_tail_mass': np.array([np.nanmean(both_SD_left_tail_mass_correlations), both_left_tail_mass_std]).tolist(),
            'both_proportion_correct': np.array([np.nanmean(both_SD_proportion_correct), both_proportion_correct_std]).tolist()
        }

        dp.dict_to_json(correlations_dict, f'../data/logpca_secondary_peak_correlations.json')
    else:
        correlations_dict = dp.json_to_dict(f'../data/logpca_secondary_peak_correlations.json')
        comp_1_sp_loc_std = correlations_dict['comp_1_SP_locs'][1]
        comp_1_left_tail_mass_std = correlations_dict['comp_1_left_tail_mass'][1]
        comp_1_proportion_correct_std = correlations_dict['comp_1_proportion_correct'][1]

        comp_2_sp_loc_std = correlations_dict['comp_2_SP_locs'][1]
        comp_2_left_tail_mass_std = correlations_dict['comp_2_left_tail_mass'][1]
        comp_2_proportion_correct_std = correlations_dict['comp_2_proportion_correct'][1]

        both_sp_loc_std = correlations_dict['both_SP_locs'][1]
        both_left_tail_mass_std = correlations_dict['both_left_tail_mass'][1]
        both_proportion_correct_std = correlations_dict['both_proportion_correct'][1]

    true_sample_metrics = get_metrics(None, emp_secondary_peak_locs = emp_secondary_peak_locs_SD, 
                                      emp_left_tail_mass = emp_left_tail_mass_SD,
                                      x_grid = x_grid, alphas = alphas, x1_ind = x1_ind)

    print(f"Comp 1 SD SP locs corr: {true_sample_metrics[0]:.3f} +- {comp_1_sp_loc_std:.3f}")
    print(f"Comp 2 SD SP locs corr: {true_sample_metrics[3]:.3f} +- {comp_2_sp_loc_std:.3f}")
    print(f"Both SD SP locs corr: {true_sample_metrics[6]:.3f} +- {both_sp_loc_std:.3f}")

    print(f"Comp 1 SD proportion correct: {true_sample_metrics[2]:.3f} +- {comp_1_proportion_correct_std:.3f}")
    print(f"Comp 2 SD proportion correct: {true_sample_metrics[5]:.3f} +- {comp_2_proportion_correct_std:.3f}")
    print(f"Both SD proportion correct: {true_sample_metrics[8]:.3f} +- {both_proportion_correct_std:.3f}")

    print(f"Comp 1 SD left tail mass corr: {true_sample_metrics[1]:.3f} +- {comp_1_left_tail_mass_std:.3f}")        
    print(f"Comp 2 SD left tail mass corr: {true_sample_metrics[4]:.3f} +- {comp_2_left_tail_mass_std:.3f}")        
    print(f"Both SD left tail mass corr: {true_sample_metrics[7]:.3f} +- {both_left_tail_mass_std:.3f}")

    import pdb; pdb.set_trace()
