import sys
sys.path.extend(['../data'])
import os
import multiprocessing as mproc
from functools import partial

import numpy as np
from scipy.stats import gaussian_kde

import data_processing_funcs as dp
from log_pca_analysis_funcs import (geodesic_quantiles_from_component, draw_samples_from_quantiles_arr, get_empirical_peaks_all, 
                                    get_empirical_peaks_all, pdf_from_quantiles_arr)

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

def get_correlations(repeat, n_samples, alphas, eps, x_grid, comp_1_projection_quantiles_BL, comp_1_projection_quantiles_SD, comp_2_projection_quantiles_BL, comp_2_projection_quantiles_SD, emp_peak_widths_BL, emp_peak_widths_SD):

    ## Comp 1 BL ##
    if not os.path.exists(f'../data/peaks/logpca_comp1_pp_width_BL_n-{n_samples}_{repeat}.npy'):
        if not os.path.exists(f'../data/peaks/logpca_comp1_pp_loc_BL_n-{n_samples}_{repeat}.npy'):
            comp_1_projection_samples_BL = draw_samples_from_quantiles_arr(comp_1_projection_quantiles_BL, alphas, n_samples=n_samples, eps=eps, 
                                                                        filedir = f'../data/samples/logpca_comp1_geodesic_projection_samples_BL_negative_xgrid_n-{n_samples}_{repeat}.npy') 
            comp_1_projection_densities_BL = pdf_from_quantiles_arr(comp_1_projection_quantiles_BL, alphas, x_grid=x_grid, samples = comp_1_projection_samples_BL, 
                                                                            filedir = f'../data/densities/logpca_comp1_geodesic_projection_densities_BL_negative_xgrid_n-{n_samples}_{repeat}.npy')[1]
            primary_peak_locs_comp_1_BL = get_empirical_peaks_all(x_grid, comp_1_projection_densities_BL, comp_1_projection_samples_BL)[0]
            np.save(f'../data/peaks/logpca_comp1_pp_loc_BL_n-{n_samples}_{repeat}.npy', primary_peak_locs_comp_1_BL)
        else:
            comp_1_projection_samples_BL = draw_samples_from_quantiles_arr(comp_1_projection_quantiles_BL, alphas, n_samples=n_samples, eps=eps, 
                                                                        filedir = f'../data/samples/logpca_comp1_geodesic_projection_samples_BL_negative_xgrid_n-{n_samples}_{repeat}.npy')
            comp_1_projection_densities_BL = pdf_from_quantiles_arr(comp_1_projection_quantiles_BL, alphas, x_grid=x_grid, samples = comp_1_projection_samples_BL, 
                                                                    filedir = f'../data/densities/logpca_comp1_geodesic_projection_densities_BL_negative_xgrid_n-{n_samples}_{repeat}.npy')[1]
            primary_peak_locs_comp_1_BL = np.load(f'../data/peaks/logpca_comp1_pp_loc_BL_n-{n_samples}_{repeat}.npy')
        peak_widths_comp_1_BL = np.array([get_peak_width(x_grid, density, peak_loc) for density, peak_loc in zip(comp_1_projection_densities_BL, primary_peak_locs_comp_1_BL)])
        np.save(f'../data/peaks/logpca_comp1_pp_width_BL_n-{n_samples}_{repeat}.npy', peak_widths_comp_1_BL)
    else:
        peak_widths_comp_1_BL = np.load(f'../data/peaks/logpca_comp1_pp_width_BL_n-{n_samples}_{repeat}.npy')
    
    comp_1_BL_correlation = np.corrcoef(peak_widths_comp_1_BL, emp_peak_widths_BL)[0, 1]

    ## Comp 1 SD ##
    if not os.path.exists(f'../data/peaks/logpca_comp1_pp_width_SD_n-{n_samples}_{repeat}.npy'):
        if not os.path.exists(f'../data/peaks/logpca_comp1_pp_loc_SD_n-{n_samples}_{repeat}.npy'):
            comp_1_projection_samples_SD = draw_samples_from_quantiles_arr(comp_1_projection_quantiles_SD, alphas, n_samples=n_samples, eps=eps, 
                                                                       filedir = f'../data/samples/logpca_comp1_geodesic_projection_samples_SD_negative_xgrid_n-{n_samples}_{repeat}.npy')
            comp_1_projection_densities_SD = pdf_from_quantiles_arr(comp_1_projection_quantiles_SD, alphas, x_grid=x_grid, samples = comp_1_projection_samples_SD,
                                                                filedir = f'../data/densities/logpca_comp1_geodesic_projection_densities_SD_negative_xgrid_n-{n_samples}_{repeat}.npy')[1]
            primary_peak_locs_comp_1_SD = get_empirical_peaks_all(x_grid, comp_1_projection_densities_SD, comp_1_projection_samples_SD)[0]
            np.save(f'../data/peaks/logpca_comp1_pp_loc_SD_n-{n_samples}_{repeat}.npy', primary_peak_locs_comp_1_SD)
        else:
            comp_1_projection_samples_SD = draw_samples_from_quantiles_arr(comp_1_projection_quantiles_SD, alphas, n_samples=n_samples, eps=eps, 
                                                                       filedir = f'../data/samples/logpca_comp1_geodesic_projection_samples_SD_negative_xgrid_n-{n_samples}_{repeat}.npy')
            comp_1_projection_densities_SD = pdf_from_quantiles_arr(comp_1_projection_quantiles_SD, alphas, x_grid=x_grid, samples = comp_1_projection_samples_SD,
                                                                filedir = f'../data/densities/logpca_comp1_geodesic_projection_densities_SD_negative_xgrid_n-{n_samples}_{repeat}.npy')[1]
            primary_peak_locs_comp_1_SD = np.load(f'../data/peaks/logpca_comp1_pp_loc_SD_n-{n_samples}_{repeat}.npy')
        peak_widths_comp_1_SD = np.array([get_peak_width(x_grid, density, peak_loc) for density, peak_loc in zip(comp_1_projection_densities_SD, primary_peak_locs_comp_1_SD)])
        np.save(f'../data/peaks/logpca_comp1_pp_width_SD_n-{n_samples}_{repeat}.npy', peak_widths_comp_1_SD)
    else:
        peak_widths_comp_1_SD = np.load(f'../data/peaks/logpca_comp1_pp_width_SD_n-{n_samples}_{repeat}.npy')

    comp_1_SD_correlation = np.corrcoef(peak_widths_comp_1_SD, emp_peak_widths_SD)[0, 1]

    ## Comp 2 BL ##
    
    if not os.path.exists(f'../data/peaks/logpca_comp2_pp_width_BL_n-{n_samples}_{repeat}.npy'):
        if not os.path.exists(f'../data/peaks/logpca_comp2_pp_loc_BL_n-{n_samples}_{repeat}.npy'):
            comp_2_projection_samples_BL = draw_samples_from_quantiles_arr(comp_2_projection_quantiles_BL, alphas, n_samples=n_samples, eps=eps, 
                                                                       filedir = f'../data/samples/logpca_comp2_geodesic_projection_samples_BL_negative_xgrid_n-{n_samples}_{repeat}.npy')
            comp_2_projection_densities_BL = pdf_from_quantiles_arr(comp_2_projection_quantiles_BL, alphas, x_grid=x_grid, samples = comp_2_projection_samples_BL,
                                                                filedir = f'../data/densities/logpca_comp2_geodesic_projection_densities_BL_negative_xgrid_n-{n_samples}_{repeat}.npy')[1]
            primary_peak_locs_comp_2_BL = get_empirical_peaks_all(x_grid, comp_2_projection_densities_BL, comp_2_projection_samples_BL)[0]
            np.save(f'../data/peaks/logpca_comp2_pp_loc_BL_n-{n_samples}_{repeat}.npy', primary_peak_locs_comp_2_BL)
        else:
            comp_2_projection_samples_BL = draw_samples_from_quantiles_arr(comp_2_projection_quantiles_BL, alphas, n_samples=n_samples, eps=eps, 
                                                                       filedir = f'../data/samples/logpca_comp2_geodesic_projection_samples_BL_negative_xgrid_n-{n_samples}_{repeat}.npy')
            comp_2_projection_densities_BL = pdf_from_quantiles_arr(comp_2_projection_quantiles_BL, alphas, x_grid=x_grid, samples = comp_2_projection_samples_BL,
                                                                filedir = f'../data/densities/logpca_comp2_geodesic_projection_densities_BL_negative_xgrid_n-{n_samples}_{repeat}.npy')[1]
            primary_peak_locs_comp_2_BL = np.load(f'../data/peaks/logpca_comp2_pp_loc_BL_n-{n_samples}_{repeat}.npy')
        peak_widths_comp_2_BL = np.array([get_peak_width(x_grid, density, peak_loc) for density, peak_loc in zip(comp_2_projection_densities_BL, primary_peak_locs_comp_2_BL)])
        np.save(f'../data/peaks/logpca_comp2_pp_width_BL_n-{n_samples}_{repeat}.npy', peak_widths_comp_2_BL)
    else:
        peak_widths_comp_2_BL = np.load(f'../data/peaks/logpca_comp2_pp_width_BL_n-{n_samples}_{repeat}.npy')

    comp_2_BL_correlation = np.corrcoef(peak_widths_comp_2_BL, emp_peak_widths_BL)[0, 1]

    ## Comp 2 SD ##
    if not os.path.exists(f'../data/peaks/logpca_comp2_pp_width_SD_n-{n_samples}_{repeat}.npy'):
        if not os.path.exists(f'../data/peaks/logpca_comp2_pp_loc_SD_n-{n_samples}_{repeat}.npy'):
            comp_2_projection_samples_SD = draw_samples_from_quantiles_arr(comp_2_projection_quantiles_SD, alphas, n_samples=n_samples, eps=eps, 
                                                                        filedir = f'../data/samples/logpca_comp2_geodesic_projection_samples_SD_negative_xgrid_n-{n_samples}_{repeat}.npy')
            comp_2_projection_densities_SD = pdf_from_quantiles_arr(comp_2_projection_quantiles_SD, alphas, x_grid=x_grid, samples = comp_2_projection_samples_SD,
                                                                    filedir = f'../data/densities/logpca_comp2_geodesic_projection_densities_SD_negative_xgrid_n-{n_samples}_{repeat}.npy')[1]
            primary_peak_locs_comp_2_SD = get_empirical_peaks_all(x_grid, comp_2_projection_densities_SD, comp_2_projection_samples_SD)[0]
            np.save(f'../data/peaks/logpca_comp2_pp_loc_SD_n-{n_samples}_{repeat}.npy', primary_peak_locs_comp_2_SD)
        else:
            comp_2_projection_samples_SD = draw_samples_from_quantiles_arr(comp_2_projection_quantiles_SD, alphas, n_samples=n_samples, eps=eps, 
                                                                        filedir = f'../data/samples/logpca_comp2_geodesic_projection_samples_SD_negative_xgrid_n-{n_samples}_{repeat}.npy')
            comp_2_projection_densities_SD = pdf_from_quantiles_arr(comp_2_projection_quantiles_SD, alphas, x_grid=x_grid, samples = comp_2_projection_samples_SD,
                                                                    filedir = f'../data/densities/logpca_comp2_geodesic_projection_densities_SD_negative_xgrid_n-{n_samples}_{repeat}.npy')[1]
            primary_peak_locs_comp_2_SD = np.load(f'../data/peaks/logpca_comp2_pp_loc_SD_n-{n_samples}_{repeat}.npy')
        peak_widths_comp_2_SD = np.array([get_peak_width(x_grid, density, peak_loc) for density, peak_loc in zip(comp_2_projection_densities_SD, primary_peak_locs_comp_2_SD)])
        np.save(f'../data/peaks/logpca_comp2_pp_width_SD_n-{n_samples}_{repeat}.npy', peak_widths_comp_2_SD)
    else:
        peak_widths_comp_2_SD = np.load(f'../data/peaks/logpca_comp2_pp_width_SD_n-{n_samples}_{repeat}.npy')

    comp_2_SD_correlation = np.corrcoef(peak_widths_comp_2_SD, emp_peak_widths_SD)[0, 1]

    return comp_1_BL_correlation, comp_1_SD_correlation, comp_2_BL_correlation, comp_2_SD_correlation

if __name__ == '__main__':

    x_grid = np.linspace(-1.0, 6.5, 2500)

    rRTs_1, rRTs_2 = dp.get_standard_individual_rRT_data()
    emp_kdes_BL = np.array([gaussian_kde(rRTs)(x_grid) for rRTs in rRTs_1])
    emp_kdes_SD = np.array([gaussian_kde(rRTs)(x_grid) for rRTs in rRTs_2])

    emp_peak_locs_BL = get_empirical_peaks_all(x_grid, emp_kdes_BL, rRTs_1)[0]
    emp_peak_locs_SD = get_empirical_peaks_all(x_grid, emp_kdes_SD, rRTs_2)[0]

    emp_peak_widths_BL = np.array([get_peak_width(x_grid, density, peak_loc) for density, peak_loc in zip(emp_kdes_BL, emp_peak_locs_BL)])
    emp_peak_widths_SD = np.array([get_peak_width(x_grid, density, peak_loc) for density, peak_loc in zip(emp_kdes_SD, emp_peak_locs_SD)])

    max_K = 8
    eps = 1e-3
    M = 200
    alphas = np.linspace(eps, 1 - eps, M)

    log_Qbar_BL = np.load(f'../data/logpca_frechet_mean_BL_{max_K}-dims.npy')
    components_BL = np.load(f'../data/logpca_components_BL_{max_K}-dims.npy')
    beta_BL = np.load(f'../data/logpca_coords_BL_{max_K}-dims.npy')

    log_Qbar_SD = np.load(f'../data/logpca_frechet_mean_SD_{max_K}-dims.npy')
    components_SD = np.load(f'../data/logpca_components_SD_{max_K}-dims.npy')
    beta_SD = np.load(f'../data/logpca_coords_SD_{max_K}-dims.npy')

    comp_1_projection_quantiles_BL = geodesic_quantiles_from_component(log_Qbar_BL, components_BL[0], t_vals=beta_BL[:, 0])[1]
    comp_1_projection_quantiles_SD = geodesic_quantiles_from_component(log_Qbar_SD, components_SD[0], t_vals=beta_SD[:, 0])[1]
    comp_2_projection_quantiles_BL = geodesic_quantiles_from_component(log_Qbar_BL, components_BL[1], t_vals=beta_BL[:, 1])[1]
    comp_2_projection_quantiles_SD = geodesic_quantiles_from_component(log_Qbar_SD, components_SD[1], t_vals=beta_SD[:, 1])[1]

    # n_repeats = 100
    n_samples = 1000
    n_repeats = 100
    ncpus = 20

    corrs_func = partial(get_correlations, n_samples = n_samples, alphas = alphas, eps = eps, x_grid = x_grid, 
                        comp_1_projection_quantiles_BL = comp_1_projection_quantiles_BL, comp_1_projection_quantiles_SD = comp_1_projection_quantiles_SD, 
                        comp_2_projection_quantiles_BL = comp_2_projection_quantiles_BL, comp_2_projection_quantiles_SD = comp_2_projection_quantiles_SD, 
                        emp_peak_widths_BL = emp_peak_widths_BL, emp_peak_widths_SD = emp_peak_widths_SD)
    
    
    with mproc.Pool(processes = ncpus) as pool:
        output = pool.map(corrs_func, np.arange(n_repeats))

    output = np.array(output)

    comp_1_BL_correlations = output[:, 0]
    comp_1_SD_correlations = output[:, 1]
    comp_2_BL_correlations = output[:, 2]
    comp_2_SD_correlations = output[:, 3]

    print(f"Comp 1 BL widths corr: {np.mean(comp_1_BL_correlations):.3f} +- {np.std(comp_1_BL_correlations):.3f}")
    print(f"Comp 1 SD widths corr: {np.mean(comp_1_SD_correlations):.3f} +- {np.std(comp_1_SD_correlations):.3f}")
    print(f"Comp 2 BL widths corr: {np.mean(comp_2_BL_correlations):.3f} +- {np.std(comp_2_BL_correlations):.3f}")
    print(f"Comp 2 SD widths corr: {np.mean(comp_2_SD_correlations):.3f} +- {np.std(comp_2_SD_correlations):.3f}")

    import pdb; pdb.set_trace()