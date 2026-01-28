import sys
sys.path.extend(['../data', '../fitting'])
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import quantile

import data_processing_funcs as dp

def geodesic_quantiles_from_component(x, v_k, t_vals=None):
    """
    Returns Q_t arrays for t in [-1,1] by default:
      Q_t = x + (t0k + t) * v_k
    """
    x = np.asarray(x)
    v_k = np.asarray(v_k)

    if t_vals is None:
        t_vals = np.linspace(-1, 1, 9)
    t_vals = np.asarray(t_vals)

    Qs = np.array([x + s * v_k for s in t_vals])
    return t_vals, Qs

def pdf_from_quantiles_kde(Q, alphas, x_grid=None, n_samples=20000, eps=1e-3):

    # sample U in (eps, 1-eps) to avoid tail craziness
    u = np.random.uniform(eps, 1 - eps, size=n_samples)
    # import pdb; pdb.set_trace()
    samples = np.interp(u, alphas, Q)

    if x_grid is None:
        lo, hi = np.quantile(samples, [0.001, 0.999])
        x_grid = np.linspace(lo, hi, 400)

    kde = gaussian_kde(samples)
    return x_grid, kde(x_grid)

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

if not os.path.exists(f'../data/logpca_comp1_geodesic_projection_densities_BL.npy'):
    import pdb; pdb.set_trace()
    comp_1_geodesic_projection_densities_BL = np.array([pdf_from_quantiles_kde(Q, alphas, x_grid=x_grid)[1] for Q in comp_1_geodesic_projection_quantiles_BL])
    np.save(f'../data/logpca_comp1_geodesic_projection_densities_BL.npy', comp_1_geodesic_projection_densities_BL)
else:
    comp_1_geodesic_projection_densities_BL = np.load(f'../data/logpca_comp1_geodesic_projection_densities_BL.npy')
if not os.path.exists(f'../data/logpca_comp2_geodesic_projection_densities_BL.npy'):
    comp_2_geodesic_projection_densities_BL = np.array([pdf_from_quantiles_kde(Q, alphas, x_grid=x_grid)[1] for Q in comp_2_geodesic_projection_quantiles_BL])
    np.save(f'../data/logpca_comp2_geodesic_projection_densities_BL.npy', comp_2_geodesic_projection_densities_BL)
else:
    comp_2_geodesic_projection_densities_BL = np.load(f'../data/logpca_comp2_geodesic_projection_densities_BL.npy')
if not os.path.exists(f'../data/logpca_comp1_geodesic_projection_densities_SD.npy'):
    comp_1_geodesic_projection_densities_SD = np.array([pdf_from_quantiles_kde(Q, alphas, x_grid=x_grid)[1] for Q in comp_1_geodesic_projection_quantiles_SD])
    np.save(f'../data/logpca_comp1_geodesic_projection_densities_SD.npy', comp_1_geodesic_projection_densities_SD)
else:
    comp_1_geodesic_projection_densities_SD = np.load(f'../data/logpca_comp1_geodesic_projection_densities_SD.npy')
if not os.path.exists(f'../data/logpca_comp2_geodesic_projection_densities_SD.npy'):
    comp_2_geodesic_projection_densities_SD = np.array([pdf_from_quantiles_kde(Q, alphas, x_grid=x_grid)[1] for Q in comp_2_geodesic_projection_quantiles_SD])
    np.save(f'../data/logpca_comp2_geodesic_projection_densities_SD.npy', comp_2_geodesic_projection_densities_SD)
else:
    comp_2_geodesic_projection_densities_SD = np.load(f'../data/logpca_comp2_geodesic_projection_densities_SD.npy')

geodesic_peak_locs_BL = np.array([x_grid[np.argmax(density)] for density in comp_1_geodesic_projection_densities_BL])
geodesic_peak_locs_SD = np.array([x_grid[np.argmax(density)] for density in comp_1_geodesic_projection_densities_SD])
comp_2_peak_locs_BL = np.array([x_grid[np.argmax(density)] for density in comp_2_geodesic_projection_densities_BL])
comp_2_peak_locs_SD = np.array([x_grid[np.argmax(density)] for density in comp_2_geodesic_projection_densities_SD])
geodesic_peak_widths_BL = np.array([get_peak_width(x_grid, density, peak_loc) for density, peak_loc in zip(comp_1_geodesic_projection_densities_BL, comp_2_peak_locs_BL)])
geodesic_peak_widths_SD = np.array([get_peak_width(x_grid, density, peak_loc) for density, peak_loc in zip(comp_1_geodesic_projection_densities_SD, comp_2_peak_locs_SD)])
comp_1_geodesic_peak_widths_BL = np.array([get_peak_width(x_grid, density, loc) for density, loc in zip(comp_1_geodesic_projection_densities_BL, geodesic_peak_locs_BL)])
comp_1_geodesic_peak_widths_SD = np.array([get_peak_width(x_grid, density, loc) for density, loc in zip(comp_1_geodesic_projection_densities_SD, geodesic_peak_locs_SD)])
# geodesic_peak_loc_range_BL = np.max(geodesic_peak_locs_BL) - np.min(geodesic_peak_locs_BL)
# geodesic_peak_loc_range_SD = np.max(geodesic_peak_locs_SD) - np.min(geodesic_peak_locs_SD)

ind_keys, pars_arr = dp.get_ind_fit_data_arr()
session_ind_arr_1 = np.arange(8)
session_ind_arr_2 = np.arange(8, 20)
RTs_dict = dp.load_RT_data_dict(raw = False, subs = None)
RTs_dict = dp.filter_data(RTs_dict, lower_cutoff = 167, upper_cutoff = 10000, 
                            remove_circ_misaligned = True)
rRTs_dict_1 = dp.get_individual_data(RTs_dict, session_ind_arr_1, scale_factor = 1000)
rRTs_dict_2 = dp.get_individual_data(RTs_dict, session_ind_arr_2, scale_factor = 1000)
rRTs_1 = [rRTs_dict_1[ind] for ind in ind_keys]
rRTs_2 = [rRTs_dict_2[ind] for ind in ind_keys]

kdes_BL = np.array([gaussian_kde(rRTs)(x_grid) for rRTs in rRTs_1])
kdes_SD = np.array([gaussian_kde(rRTs)(x_grid) for rRTs in rRTs_2])

emp_peak_locs_BL = np.array([x_grid[np.argmax(kde)] for kde in kdes_BL])
emp_peak_locs_SD = np.array([x_grid[np.argmax(kde)] for kde in kdes_SD])
emp_peak_widths_BL = np.array([get_peak_width(x_grid, kde, peak_loc) for kde, peak_loc in zip(kdes_BL, emp_peak_locs_BL)])
emp_peak_widths_SD = np.array([get_peak_width(x_grid, kde, peak_loc) for kde, peak_loc in zip(kdes_SD, emp_peak_locs_SD)])
emp_peak_loc_range_BL = np.max(emp_peak_locs_BL) - np.min(emp_peak_locs_BL)
emp_peak_loc_range_SD = np.max(emp_peak_locs_SD) - np.min(emp_peak_locs_SD)

BL_locs_corr = np.corrcoef(geodesic_peak_locs_BL, emp_peak_locs_BL)[0, 1]
SD_locs_corr = np.corrcoef(geodesic_peak_locs_SD, emp_peak_locs_SD)[0, 1]
BL_widths_corr = np.corrcoef(geodesic_peak_widths_BL, emp_peak_widths_BL)[0, 1]
SD_widths_corr = np.corrcoef(geodesic_peak_widths_SD, emp_peak_widths_SD)[0, 1]

BL_locs_corr_comp2 = np.corrcoef(comp_2_peak_locs_BL, emp_peak_locs_BL)[0, 1]
SD_locs_corr_comp2 = np.corrcoef(comp_2_peak_locs_SD, emp_peak_locs_SD)[0, 1]
BL_widths_corr_comp1 = np.corrcoef(comp_1_geodesic_peak_widths_BL, emp_peak_widths_BL)[0, 1]
SD_widths_corr_comp1 = np.corrcoef(comp_1_geodesic_peak_widths_SD, emp_peak_widths_SD)[0, 1]

plt.scatter(geodesic_peak_locs_BL, emp_peak_locs_BL)
plt.show()
plt.scatter(geodesic_peak_locs_SD, emp_peak_locs_SD)
plt.show()
plt.scatter(geodesic_peak_widths_BL, emp_peak_widths_BL)
plt.show()
plt.scatter(geodesic_peak_widths_SD, emp_peak_widths_SD)
plt.show()

# BL_peak_range_ratio = geodesic_peak_loc_range_BL / emp_peak_loc_range_BL
# SD_peak_range_ratio = geodesic_peak_loc_range_SD / emp_peak_loc_range_SD

import pdb; pdb.set_trace()

# peak_loc_range_BL = np.array([np.min(emp_peak_locs_BL), np.max(emp_peak_locs_BL)])


plt.scatter(peak_locs_BL, peak_locs_SD)
plt.show()