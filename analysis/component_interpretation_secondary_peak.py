## TODO: Add repeat sampling for uncertainty estimation ##

import sys
sys.path.extend(['../data'])
import os

import numpy as np
from scipy.stats import gaussian_kde

import data_processing_funcs as dp
from log_pca_analysis_funcs import (geodesic_quantiles_from_component, pdf_from_quantiles_arr, draw_samples_from_quantiles_arr, 
                                    get_empirical_peaks_all, reconstruct_quantiles)

x_grid = np.linspace(-1.0, 6.5, 2500)
x1_ind = np.where(x_grid >= 1.0)[0][0]
dx = x_grid[1] - x_grid[0]

_, rRTs_2 = dp.get_standard_individual_rRT_data()
emp_kdes_SD = np.array([gaussian_kde(rRTs)(x_grid) for rRTs in rRTs_2])
emp_secondary_peak_locs_SD = get_empirical_peaks_all(x_grid, emp_kdes_SD, rRTs_2)[1]

emp_left_tail_mass_SD = np.array([np.trapz(kde[:x1_ind], x_grid[:x1_ind]) for kde in emp_kdes_SD])

max_K = 8
eps = 1e-3
M = 200
alphas = np.linspace(eps, 1 - eps, M)

log_Qbar_SD = np.load(f'../data/logpca_frechet_mean_SD_{max_K}-dims.npy')
components_SD = np.load(f'../data/logpca_components_SD_{max_K}-dims.npy')
beta_SD = np.load(f'../data/logpca_coords_SD_{max_K}-dims.npy')
explained_variance_SD = np.load(f'../data/logpca_explained_variance_SD_{max_K}-dims.npy')

comp_1_projection_quantiles_SD = geodesic_quantiles_from_component(log_Qbar_SD, components_SD[0], t_vals=beta_SD[:, 0])[1]
comp_2_projection_quantiles_SD = geodesic_quantiles_from_component(log_Qbar_SD, components_SD[1], t_vals=beta_SD[:, 1])[1]
both_components_projection_quantiles_SD = reconstruct_quantiles(log_Qbar_SD, components_SD[:2], beta_SD[:, :2])

n_samples = 1000
n_repeats = 1

comp_1_SD_sp_loc_correlations = np.zeros(n_repeats)
comp_1_SD_proportion_correct = np.zeros(n_repeats)
comp_1_SD_left_tail_mass_correlations = np.zeros(n_repeats)

comp_2_SD_sp_loc_correlations = np.zeros(n_repeats)
comp_2_SD_proportion_correct = np.zeros(n_repeats)
comp_2_SD_left_tail_mass_correlations = np.zeros(n_repeats)

both_SD_sp_loc_correlations = np.zeros(n_repeats)
both_SD_proportion_correct = np.zeros(n_repeats)
both_SD_left_tail_mass_correlations = np.zeros(n_repeats)

for repeat in range(n_repeats):

    ## Comp 1 SD ##
    if not os.path.exists(f'../data/peaks/logpca_comp1_sp_loc_SD_n-{n_samples}_{repeat}.npy'):
        comp_1_projection_samples_SD = draw_samples_from_quantiles_arr(comp_1_projection_quantiles_SD, alphas, n_samples=n_samples, eps=eps, 
                                                                       filedir = f'../data/samples/logpca_comp1_geodesic_projection_samples_SD_negative_xgrid_n-{n_samples}_{repeat}.npy')
        comp_1_projection_densities_SD = pdf_from_quantiles_arr(comp_1_projection_quantiles_SD, alphas, x_grid=x_grid, samples = comp_1_projection_samples_SD,
                                                                filedir = f'../data/densities/logpca_comp1_geodesic_projection_densities_SD_negative_xgrid_n-{n_samples}_{repeat}.npy')[1]
        secondary_peak_locs_comp_1_SD = get_empirical_peaks_all(x_grid, comp_1_projection_densities_SD, comp_1_projection_samples_SD)[1]
        np.save(f'../data/peaks/logpca_comp1_sp_loc_SD_n-{n_samples}_{repeat}.npy', secondary_peak_locs_comp_1_SD)
    else:
        comp_1_projection_samples_SD = draw_samples_from_quantiles_arr(comp_1_projection_quantiles_SD, alphas, n_samples=n_samples, eps=eps, 
                                                                       filedir = f'../data/samples/logpca_comp1_geodesic_projection_samples_SD_negative_xgrid_n-{n_samples}_{repeat}.npy')
        comp_1_projection_densities_SD = pdf_from_quantiles_arr(comp_1_projection_quantiles_SD, alphas, x_grid=x_grid, samples = comp_1_projection_samples_SD,
                                                                filedir = f'../data/densities/logpca_comp1_geodesic_projection_densities_SD_negative_xgrid_n-{n_samples}_{repeat}.npy')[1]
        secondary_peak_locs_comp_1_SD = np.load(f'../data/peaks/logpca_comp1_sp_loc_SD_n-{n_samples}_{repeat}.npy')
    
    comp_1_correctly_has_sp = (~np.isnan(emp_secondary_peak_locs_SD))*(~np.isnan(secondary_peak_locs_comp_1_SD))
    comp_1_proportion_correct = np.sum(comp_1_correctly_has_sp) / len(comp_1_correctly_has_sp)
    comp_1_correct_sp_inds = np.nonzero(comp_1_correctly_has_sp)[0]
    comp_1_left_tail_mass_SD = np.array([np.trapz(dens[:x1_ind], x_grid[:x1_ind]) for dens in comp_1_projection_densities_SD])
    comp_1_SD_sp_loc_correlations[repeat] = np.corrcoef(secondary_peak_locs_comp_1_SD[comp_1_correct_sp_inds], emp_secondary_peak_locs_SD[comp_1_correct_sp_inds])[0, 1]
    comp_1_SD_left_tail_mass_correlations[repeat] = np.corrcoef(comp_1_left_tail_mass_SD[comp_1_correct_sp_inds], emp_left_tail_mass_SD[comp_1_correct_sp_inds])[0, 1]
    comp_1_SD_proportion_correct[repeat] = comp_1_proportion_correct

    ## Comp 2 SD ##
    if not os.path.exists(f'../data/peaks/logpca_comp2_sp_loc_SD_n-{n_samples}_{repeat}.npy'):
        comp_2_projection_samples_SD = draw_samples_from_quantiles_arr(comp_2_projection_quantiles_SD, alphas, n_samples=n_samples, eps=eps, 
                                                                       filedir = f'../data/samples/logpca_comp2_geodesic_projection_samples_SD_negative_xgrid_n-{n_samples}_{repeat}.npy')
        comp_2_projection_densities_SD = pdf_from_quantiles_arr(comp_2_projection_quantiles_SD, alphas, x_grid=x_grid, samples = comp_2_projection_samples_SD,
                                                                filedir = f'../data/densities/logpca_comp2_geodesic_projection_densities_SD_negative_xgrid_n-{n_samples}_{repeat}.npy')[1]
        secondary_peak_locs_comp_2_SD = get_empirical_peaks_all(x_grid, comp_2_projection_densities_SD, comp_2_projection_samples_SD)[1]
        np.save(f'../data/peaks/logpca_comp2_sp_loc_SD_n-{n_samples}_{repeat}.npy', secondary_peak_locs_comp_2_SD)
    else:
        comp_2_projection_samples_SD = draw_samples_from_quantiles_arr(comp_2_projection_quantiles_SD, alphas, n_samples=n_samples, eps=eps, 
                                                                       filedir = f'../data/samples/logpca_comp2_geodesic_projection_samples_SD_negative_xgrid_n-{n_samples}_{repeat}.npy')
        comp_2_projection_densities_SD = pdf_from_quantiles_arr(comp_2_projection_quantiles_SD, alphas, x_grid=x_grid, samples = comp_2_projection_samples_SD,
                                                                filedir = f'../data/densities/logpca_comp2_geodesic_projection_densities_SD_negative_xgrid_n-{n_samples}_{repeat}.npy')[1]
        secondary_peak_locs_comp_2_SD = np.load(f'../data/peaks/logpca_comp2_sp_loc_SD_n-{n_samples}_{repeat}.npy')

    comp_2_correctly_has_sp = (~np.isnan(emp_secondary_peak_locs_SD))*(~np.isnan(secondary_peak_locs_comp_2_SD))
    comp_2_proportion_correct = np.sum(comp_2_correctly_has_sp) / len(comp_2_correctly_has_sp)
    comp_2_correct_sp_inds = np.nonzero(comp_2_correctly_has_sp)[0]
    comp_2_left_tail_mass_SD = np.array([np.trapz(dens[:x1_ind], x_grid[:x1_ind]) for dens in comp_2_projection_densities_SD])
    comp_2_SD_sp_loc_correlations[repeat] = np.corrcoef(secondary_peak_locs_comp_2_SD[comp_2_correct_sp_inds], emp_secondary_peak_locs_SD[comp_2_correct_sp_inds])[0, 1]
    comp_2_SD_left_tail_mass_correlations[repeat] = np.corrcoef(comp_2_left_tail_mass_SD[comp_2_correct_sp_inds], emp_left_tail_mass_SD[comp_2_correct_sp_inds])[0, 1]
    comp_2_SD_proportion_correct[repeat] = comp_2_proportion_correct

    ## Both Comps SD ##
    if not os.path.exists(f'../data/peaks/logpca_both_sp_loc_SD_n-{n_samples}_{repeat}.npy'):
        both_projection_samples_SD = draw_samples_from_quantiles_arr(both_components_projection_quantiles_SD, alphas, n_samples=n_samples, eps=eps, 
                                                                     filedir = f'../data/samples/logpca_both_geodesic_projection_samples_SD_negative_xgrid_n-{n_samples}_{repeat}.npy')
        both_projection_densities_SD = pdf_from_quantiles_arr(both_components_projection_quantiles_SD, alphas, x_grid=x_grid, samples = both_projection_samples_SD,
                                                              filedir = f'../data/densities/logpca_both_geodesic_projection_densities_SD_negative_xgrid_n-{n_samples}_{repeat}.npy')[1]
        secondary_peak_locs_both_SD = get_empirical_peaks_all(x_grid, both_projection_densities_SD, both_projection_samples_SD)[1]
        np.save(f'../data/peaks/logpca_both_sp_loc_SD_n-{n_samples}_{repeat}.npy', secondary_peak_locs_both_SD)
    else:
        both_projection_samples_SD = draw_samples_from_quantiles_arr(both_components_projection_quantiles_SD, alphas, n_samples=n_samples, eps=eps, 
                                                                     filedir = f'../data/samples/logpca_both_geodesic_projection_samples_SD_negative_xgrid_n-{n_samples}_{repeat}.npy')
        both_projection_densities_SD = pdf_from_quantiles_arr(both_components_projection_quantiles_SD, alphas, x_grid=x_grid, samples = both_projection_samples_SD,
                                                              filedir = f'../data/densities/logpca_both_geodesic_projection_densities_SD_negative_xgrid_n-{n_samples}_{repeat}.npy')[1]
        secondary_peak_locs_both_SD = np.load(f'../data/peaks/logpca_both_sp_loc_SD_n-{n_samples}_{repeat}.npy')

    both_correctly_has_sp = (~np.isnan(emp_secondary_peak_locs_SD))*(~np.isnan(secondary_peak_locs_both_SD))
    both_proportion_correct = np.sum(both_correctly_has_sp) / len(both_correctly_has_sp)
    both_correct_sp_inds = np.nonzero(both_correctly_has_sp)[0]
    both_left_tail_mass_SD = np.array([np.trapz(dens[:x1_ind], x_grid[:x1_ind]) for dens in both_projection_densities_SD])
    both_SD_sp_loc_correlations[repeat] = np.corrcoef(secondary_peak_locs_both_SD[both_correct_sp_inds], emp_secondary_peak_locs_SD[both_correct_sp_inds])[0, 1]
    both_SD_left_tail_mass_correlations[repeat] = np.corrcoef(both_left_tail_mass_SD[both_correct_sp_inds], emp_left_tail_mass_SD[both_correct_sp_inds])[0, 1]
    both_SD_proportion_correct[repeat] = both_proportion_correct

print(f"Comp 1 SD SP locs corr: {np.mean(comp_1_SD_sp_loc_correlations):.3f} +- {np.std(comp_1_SD_sp_loc_correlations):.3f}")
print(f"Comp 2 SD SP locs corr: {np.mean(comp_2_SD_sp_loc_correlations):.3f} +- {np.std(comp_2_SD_sp_loc_correlations):.3f}")
print(f"Both SD SP locs corr: {np.mean(both_SD_sp_loc_correlations):.3f} +- {np.std(both_SD_sp_loc_correlations):.3f}")

print(f"Comp 1 SD proportion correct: {np.mean(comp_1_SD_proportion_correct):.3f} +- {np.std(comp_1_SD_proportion_correct):.3f}")
print(f"Comp 2 SD proportion correct: {np.mean(comp_2_SD_proportion_correct):.3f} +- {np.std(comp_2_SD_proportion_correct):.3f}")
print(f"Both SD proportion correct: {np.mean(both_SD_proportion_correct):.3f} +- {np.std(both_SD_proportion_correct):.3f}")

print(f"Comp 1 SD left tail mass corr: {np.mean(comp_1_SD_left_tail_mass_correlations):.3f} +- {np.std(comp_1_SD_left_tail_mass_correlations):.3f}")        
print(f"Comp 2 SD left tail mass corr: {np.mean(comp_2_SD_left_tail_mass_correlations):.3f} +- {np.std(comp_2_SD_left_tail_mass_correlations):.3f}")        
print(f"Both SD left tail mass corr: {np.mean(both_SD_left_tail_mass_correlations):.3f} +- {np.std(both_SD_left_tail_mass_correlations):.3f}")
# for K in range(2, max_K):
#     sp_locs_K = sp_locs_3_up[K-2]
#     sp_inds_K = np.nonzero((~np.isnan(emp_secondary_peak_locs_SD))*(~np.isnan(sp_locs_K)))[0]
#     sp_gets_K = np.sum((~np.isnan(emp_secondary_peak_locs_SD))*(~np.isnan(sp_locs_K)))/len(rRTs_SD)
#     sp_corr_locs_K = np.corrcoef(sp_locs_K[sp_inds_K], emp_secondary_peak_locs_SD[sp_inds_K])[0, 1]
#     proportion_correct_3_up[K-2] = sp_gets_K
#     correlation_locs_3_up[K-2] = sp_corr_locs_K
#     print(f"For K={K+1}, proportion getting secondary peak presence correct: {sp_gets_K}, correlation of secondary peak locations: {sp_corr_locs_K}")



import pdb; pdb.set_trace()
