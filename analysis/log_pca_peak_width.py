import sys
sys.path.extend(['../data'])

import numpy as np
from scipy.stats import gaussian_kde

import data_processing_funcs as dp
from log_pca_analysis_funcs import geodesic_quantiles_from_component, pdf_from_quantiles_arr

def get_peak_width(x_grid, density, peak_loc, height_frac=0.5):
    peak_height = np.max(density)
    half_height = peak_height * height_frac

    # Find left width
    left_indices = np.where(x_grid < peak_loc)[0]
    left_densities = density[left_indices]
    left_x = x_grid[left_indices]

    left_crossings = np.where(left_densities <= half_height)[0]
    if len(left_crossings) == 0:
        left_width = peak_loc - x_grid[0]
    else:
        left_idx = left_crossings[-1]
        left_width = peak_loc - left_x[left_idx]

    # Find right width
    right_indices = np.where(x_grid > peak_loc)[0]
    right_densities = density[right_indices]
    right_x = x_grid[right_indices]

    right_crossings = np.where(right_densities <= half_height)[0]
    if len(right_crossings) == 0:
        right_width = x_grid[-1] - peak_loc
    else:
        right_idx = right_crossings[0]
        right_width = right_x[right_idx] - peak_loc

    total_width = left_width + right_width
    return total_width

max_K = 8
x_grid = np.linspace(0.01, 6.5, 2000)
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


comp_1_geodesic_projection_densities_BL = pdf_from_quantiles_arr(comp_1_geodesic_projection_quantiles_BL, alphas, x_grid=x_grid, filedir = f'../data/logpca_comp1_geodesic_projection_densities_BL.npy')[1]
comp_2_geodesic_projection_densities_BL = pdf_from_quantiles_arr(comp_2_geodesic_projection_quantiles_BL, alphas, x_grid=x_grid, filedir = f'../data/logpca_comp2_geodesic_projection_densities_BL.npy')[1]
comp_1_geodesic_projection_densities_SD = pdf_from_quantiles_arr(comp_1_geodesic_projection_quantiles_SD, alphas, x_grid=x_grid, filedir = f'../data/logpca_comp1_geodesic_projection_densities_SD.npy')[1]
comp_2_geodesic_projection_densities_SD = pdf_from_quantiles_arr(comp_2_geodesic_projection_quantiles_SD, alphas, x_grid=x_grid, filedir = f'../data/logpca_comp2_geodesic_projection_densities_SD.npy')[1]

comp_1_peak_locs_BL = np.array([x_grid[np.argmax(density)] for density in comp_1_geodesic_projection_densities_BL])
comp_1_peak_locs_SD = np.array([x_grid[np.argmax(density)] for density in comp_1_geodesic_projection_densities_SD])
comp_2_peak_locs_BL = np.array([x_grid[np.argmax(density)] for density in comp_2_geodesic_projection_densities_BL])
comp_2_peak_locs_SD = np.array([x_grid[np.argmax(density)] for density in comp_2_geodesic_projection_densities_SD])
geodesic_peak_widths_BL = np.array([get_peak_width(x_grid, density, peak_loc) for density, peak_loc in zip(comp_2_geodesic_projection_densities_BL, comp_2_peak_locs_BL)])
geodesic_peak_widths_SD = np.array([get_peak_width(x_grid, density, peak_loc) for density, peak_loc in zip(comp_2_geodesic_projection_densities_SD, comp_2_peak_locs_SD)])
comp_1_geodesic_peak_widths_BL = np.array([get_peak_width(x_grid, density, loc) for density, loc in zip(comp_1_geodesic_projection_densities_BL, comp_1_peak_locs_BL)])
comp_1_geodesic_peak_widths_SD = np.array([get_peak_width(x_grid, density, loc) for density, loc in zip(comp_1_geodesic_projection_densities_SD, comp_1_peak_locs_SD)])


rRTs_1, rRTs_2 = dp.get_standard_individual_rRT_data()

kdes_BL = np.array([gaussian_kde(rRTs)(x_grid) for rRTs in rRTs_1])
kdes_SD = np.array([gaussian_kde(rRTs)(x_grid) for rRTs in rRTs_2])

emp_peak_locs_BL = np.array([x_grid[np.argmax(kde)] for kde in kdes_BL])
emp_peak_locs_SD = np.array([x_grid[np.argmax(kde)] for kde in kdes_SD])
emp_peak_widths_BL = np.array([get_peak_width(x_grid, kde, peak_loc) for kde, peak_loc in zip(kdes_BL, emp_peak_locs_BL)])
emp_peak_widths_SD = np.array([get_peak_width(x_grid, kde, peak_loc) for kde, peak_loc in zip(kdes_SD, emp_peak_locs_SD)])

BL_widths_corr = np.corrcoef(geodesic_peak_widths_BL, emp_peak_widths_BL)[0, 1]
SD_widths_corr = np.corrcoef(geodesic_peak_widths_SD, emp_peak_widths_SD)[0, 1]

BL_widths_corr_comp1 = np.corrcoef(comp_1_geodesic_peak_widths_BL, emp_peak_widths_BL)[0, 1]
SD_widths_corr_comp1 = np.corrcoef(comp_1_geodesic_peak_widths_SD, emp_peak_widths_SD)[0, 1]

import pdb; pdb.set_trace()

print(f"BL widths corr: {BL_widths_corr:.3f}, SD widths corr: {SD_widths_corr:.3f}")
print(f"BL widths corr comp1: {BL_widths_corr_comp1:.3f}, SD widths corr comp1: {SD_widths_corr_comp1:.3f}")