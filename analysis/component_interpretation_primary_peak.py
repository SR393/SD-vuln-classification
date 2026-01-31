import sys
sys.path.extend(['../data'])

import numpy as np
from scipy.stats import gaussian_kde

import data_processing_funcs as dp
from log_pca_analysis_funcs import geodesic_quantiles_from_component, pdf_from_quantiles_arr

max_K = 8
x_grid = np.linspace(-1.0, 6.5, 2500)
eps = 1e-3
M = 200
alphas = np.linspace(eps, 1 - eps, M)

log_Qbar_BL = np.load(f'../data/logpca_frechet_mean_BL_{max_K}-dims.npy')
components_BL = np.load(f'../data/logpca_components_BL_{max_K}-dims.npy')
beta_BL = np.load(f'../data/logpca_coords_BL_{max_K}-dims.npy')

log_Qbar_SD = np.load(f'../data/logpca_frechet_mean_SD_{max_K}-dims.npy')
components_SD = np.load(f'../data/logpca_components_SD_{max_K}-dims.npy')
beta_SD = np.load(f'../data/logpca_coords_SD_{max_K}-dims.npy')

comp_1_geodesic_projection_quantiles_BL = geodesic_quantiles_from_component(log_Qbar_BL, components_BL[0], t_vals=beta_BL[:, 0])[1]
comp_2_geodesic_projection_quantiles_BL = geodesic_quantiles_from_component(log_Qbar_BL, components_BL[1], t_vals=beta_BL[:, 1])[1]
comp_1_geodesic_projection_quantiles_SD = geodesic_quantiles_from_component(log_Qbar_SD, components_SD[0], t_vals=beta_SD[:, 0])[1]
comp_2_geodesic_projection_quantiles_SD = geodesic_quantiles_from_component(log_Qbar_SD, components_SD[1], t_vals=beta_SD[:, 1])[1]

comp_1_geodesic_projection_densities_BL = pdf_from_quantiles_arr(comp_1_geodesic_projection_quantiles_BL, alphas, x_grid=x_grid, filedir = f'../data/logpca_comp1_geodesic_projection_densities_BL_negative_xgrid.npy')[1]
comp_2_geodesic_projection_densities_BL = pdf_from_quantiles_arr(comp_2_geodesic_projection_quantiles_BL, alphas, x_grid=x_grid, filedir = f'../data/logpca_comp2_geodesic_projection_densities_BL_negative_xgrid.npy')[1]
comp_1_geodesic_projection_densities_SD = pdf_from_quantiles_arr(comp_1_geodesic_projection_quantiles_SD, alphas, x_grid=x_grid, filedir = f'../data/logpca_comp1_geodesic_projection_densities_SD_negative_xgrid.npy')[1]
comp_2_geodesic_projection_densities_SD = pdf_from_quantiles_arr(comp_2_geodesic_projection_quantiles_SD, alphas, x_grid=x_grid, filedir = f'../data/logpca_comp2_geodesic_projection_densities_SD_negative_xgrid.npy')[1]

import pdb; pdb.set_trace()

geodesic_peak_locs_BL = np.array([x_grid[np.argmax(density)] for density in comp_1_geodesic_projection_densities_BL])
geodesic_peak_locs_SD = np.array([x_grid[np.argmax(density)] for density in comp_1_geodesic_projection_densities_SD])
comp_2_peak_locs_BL = np.array([x_grid[np.argmax(density)] for density in comp_2_geodesic_projection_densities_BL])
comp_2_peak_locs_SD = np.array([x_grid[np.argmax(density)] for density in comp_2_geodesic_projection_densities_SD])

rRTs_1, rRTs_2 = dp.get_standard_individual_rRT_data()

kdes_BL = np.array([gaussian_kde(rRTs)(x_grid) for rRTs in rRTs_1])
kdes_SD = np.array([gaussian_kde(rRTs)(x_grid) for rRTs in rRTs_2])

emp_peak_locs_BL = np.array([x_grid[np.argmax(kde)] for kde in kdes_BL])
emp_peak_locs_SD = np.array([x_grid[np.argmax(kde)] for kde in kdes_SD])

BL_locs_corr = np.corrcoef(geodesic_peak_locs_BL, emp_peak_locs_BL)[0, 1]
SD_locs_corr = np.corrcoef(geodesic_peak_locs_SD, emp_peak_locs_SD)[0, 1]

BL_locs_corr_comp2 = np.corrcoef(comp_2_peak_locs_BL, emp_peak_locs_BL)[0, 1]
SD_locs_corr_comp2 = np.corrcoef(comp_2_peak_locs_SD, emp_peak_locs_SD)[0, 1]

print(f"BL locs corr: {BL_locs_corr:.3f}, SD locs corr: {SD_locs_corr:.3f}")
print(f"BL locs corr comp2: {BL_locs_corr_comp2:.3f}, SD locs corr comp2: {SD_locs_corr_comp2:.3f}")