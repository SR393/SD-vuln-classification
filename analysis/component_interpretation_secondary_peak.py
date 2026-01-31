import sys
sys.path.extend(['../data'])

import numpy as np
from scipy.stats import gaussian_kde

import data_processing_funcs as dp
from log_pca_analysis_funcs import (geodesic_quantiles_from_component, pdf_from_quantiles_arr, draw_samples_from_quantiles, 
                                    get_empirical_peaks_all, reconstruct_quantiles)

def get_secondary_peak_location(density, x_grid=None):
    """
    density: (N, L) array of densities on x_grid
    returns: (N,) array of secondary peak locations
    """
    if x_grid is None:
        x_grid = np.linspace(0.01, 6.5, 2000)
    zeros = dp.get_derivative_zeros(x_grid, density)
    if len(zeros) == 1:
        return np.nan
    else:
        return zeros[0]

max_K = 8
x_grid = np.linspace(-1.0, 6.5, 2500)
eps = 1e-3
M = 200
alphas = np.linspace(eps, 1 - eps, M)

log_Qbar_SD = np.load(f'../data/logpca_frechet_mean_SD_{max_K}-dims.npy')
components_SD = np.load(f'../data/logpca_components_SD_{max_K}-dims.npy')
beta_SD = np.load(f'../data/logpca_coords_SD_{max_K}-dims.npy')
explained_variance_SD = np.load(f'../data/logpca_explained_variance_SD_{max_K}-dims.npy')

comp_1_geodesic_projection_quantiles_SD = geodesic_quantiles_from_component(log_Qbar_SD, components_SD[0], t_vals=beta_SD[:, 0])[1]
# import pdb; pdb.set_trace()
comp_1_projection_samples = np.array([draw_samples_from_quantiles(Q, alphas, n_samples=1000, eps=eps) for Q in comp_1_geodesic_projection_quantiles_SD]) 
comp_1_geodesic_projection_densities_SD = pdf_from_quantiles_arr(comp_1_geodesic_projection_quantiles_SD, alphas, x_grid=x_grid, 
                                                                 samples = comp_1_projection_samples, filedir = f'../data/logpca_comp1_geodesic_projection_densities_SD_negative_xgrid_n-1000.npy')[1]

comp_2_geodesic_projection_quantiles_SD = geodesic_quantiles_from_component(log_Qbar_SD, components_SD[1], t_vals=beta_SD[:, 1])[1]
comp_2_projection_samples = np.array([draw_samples_from_quantiles(Q, alphas, n_samples=1000, eps=eps) for Q in comp_2_geodesic_projection_quantiles_SD]) 

comp_2_geodesic_projection_densities_SD = pdf_from_quantiles_arr(comp_2_geodesic_projection_quantiles_SD, alphas, x_grid=x_grid, 
                                                                 samples = comp_2_projection_samples, filedir = f'../data/logpca_comp2_geodesic_projection_densities_SD_negative_xgrid_n-1000.npy')[1]

both_components_projection_quantiles = reconstruct_quantiles(log_Qbar_SD, components_SD[:2], beta_SD[:, :2])
both_components_projection_samples = np.array([draw_samples_from_quantiles(Q, alphas, n_samples=1000, eps=eps) for Q in both_components_projection_quantiles])
both_components_projection_densities = pdf_from_quantiles_arr(both_components_projection_quantiles, alphas, x_grid=x_grid, 
                                                             samples = both_components_projection_samples, filedir = f'../data/logpca_2_components_geodesic_projection_densities_SD_negative_xgrid_n-1000.npy')[1]

primary_peak_locs_comp_1_SD, secondary_peak_locs_comp_1_SD, primary_peak_proportions_comp_1_SD, secondary_peak_proportions_comp_1_SD = get_empirical_peaks_all(x_grid, comp_1_geodesic_projection_densities_SD, comp_1_projection_samples)
primary_peak_locs_comp_2_SD, secondary_peak_locs_comp_2_SD, primary_peak_proportions_comp_2_SD, secondary_peak_proportions_comp_2_SD = get_empirical_peaks_all(x_grid, comp_2_geodesic_projection_densities_SD, comp_2_projection_samples)
primary_peak_locs_both_SD, secondary_peak_locs_both_SD, primary_peak_proportions_both_SD, secondary_peak_proportions_both_SD = get_empirical_peaks_all(x_grid, both_components_projection_densities, both_components_projection_samples)

_, rRTs_SD = dp.get_standard_individual_rRT_data()
kdes_SD = np.array([gaussian_kde(rRTs)(x_grid) for rRTs in rRTs_SD])
emp_primary_peak_locs_SD, emp_secondary_peak_locs_SD, emp_primary_peak_proportions_SD, emp_secondary_peak_proportions_SD = get_empirical_peaks_all(x_grid, kdes_SD, rRTs_SD)

comp_1_sp = (~np.isnan(emp_secondary_peak_locs_SD))*(~np.isnan(secondary_peak_locs_comp_1_SD))
comp_2_sp = (~np.isnan(emp_secondary_peak_locs_SD))*(~np.isnan(secondary_peak_locs_comp_2_SD))
both_sp = (~np.isnan(emp_secondary_peak_locs_SD))*(~np.isnan(secondary_peak_locs_both_SD))
comp_1_nsp = (np.isnan(emp_secondary_peak_locs_SD))*(np.isnan(secondary_peak_locs_comp_1_SD))
comp_2_nsp = (np.isnan(emp_secondary_peak_locs_SD))*(np.isnan(secondary_peak_locs_comp_2_SD))
both_nsp = (np.isnan(emp_secondary_peak_locs_SD))*(np.isnan(secondary_peak_locs_both_SD))

comp_1_inds_sp = np.nonzero(comp_1_sp)[0]
comp_2_inds_sp = np.nonzero(comp_2_sp)[0]
both_inds_sp = np.nonzero(both_sp)[0]
comp_1_inds_nsp = np.nonzero(comp_1_nsp)[0]
comp_2_inds_nsp = np.nonzero(comp_2_nsp)[0]
both_inds_nsp = np.nonzero(both_nsp)[0]

# import pdb; pdb.set_trace()

comp_1_gets_sp = np.sum(comp_1_sp + comp_1_nsp)/len(rRTs_SD)
comp_2_gets_sp = np.sum(comp_2_sp + comp_2_nsp)/len(rRTs_SD)
both_gets_sp = np.sum(both_sp + both_nsp)/len(rRTs_SD)

comp_1_corr_sp_locs = np.corrcoef(secondary_peak_locs_comp_1_SD[comp_1_inds_sp], emp_secondary_peak_locs_SD[comp_1_inds_sp])[0, 1]
comp_2_corr_sp_locs = np.corrcoef(secondary_peak_locs_comp_2_SD[comp_2_inds_sp], emp_secondary_peak_locs_SD[comp_2_inds_sp])[0, 1]
both_corr_sp_locs = np.corrcoef(secondary_peak_locs_both_SD[both_inds_sp], emp_secondary_peak_locs_SD[both_inds_sp])[0, 1]

print("Proportion of comp 1 projections that get secondary peak presence correct:", comp_1_gets_sp)
print("Proportion of comp 2 projections that get secondary peak presence correct:", comp_2_gets_sp)
print("Proportion of both components projections that get secondary peak presence correct:", both_gets_sp)

print("Correlation of secondary peak locations for comp 1 projections:", comp_1_corr_sp_locs)
print("Correlation of secondary peak locations for comp 2 projections:", comp_2_corr_sp_locs)
print("Correlation of secondary peak locations for both components projections:", both_corr_sp_locs)

import pdb; pdb.set_trace()
