import sys
sys.path.extend(['../data'])

import numpy as np
from scipy.stats import gaussian_kde

import data_processing_funcs as dp
from log_pca_analysis_funcs import geodesic_quantiles_from_component, pdf_from_quantiles_kde

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
x_grid = np.linspace(-0.5, 6.5, 2000)
eps = 1e-3
M = 200
alphas = np.linspace(eps, 1 - eps, M)

log_Qbar_SD = np.load(f'../data/logpca_frechet_mean_SD_{max_K}-dims.npy')
components_SD = np.load(f'../data/logpca_components_SD_{max_K}-dims.npy')
explained_variance_SD = np.load(f'../data/logpca_explained_variance_SD_{max_K}-dims.npy')

comp_1_coord = -2*np.sqrt(explained_variance_SD[0])
comp_1_geodesic_projection_quantiles_SD = geodesic_quantiles_from_component(log_Qbar_SD, components_SD[0], t_vals=[comp_1_coord])[1][0]
comp_1_geodesic_projection_densities_SD = pdf_from_quantiles_kde(comp_1_geodesic_projection_quantiles_SD, alphas, x_grid=x_grid)[1]
comp_1_secondary_peak_loc_SD = get_secondary_peak_location(comp_1_geodesic_projection_densities_SD, x_grid=x_grid)

comp_2_coord = -2*np.sqrt(explained_variance_SD[1])
comp_2_geodesic_projection_quantiles_SD = geodesic_quantiles_from_component(log_Qbar_SD, components_SD[1], t_vals=[comp_2_coord])[1][0]
comp_2_geodesic_projection_densities_SD = pdf_from_quantiles_kde(comp_2_geodesic_projection_quantiles_SD, alphas, x_grid=x_grid)[1]
comp_2_secondary_peak_loc_SD = get_secondary_peak_location(comp_2_geodesic_projection_densities_SD, x_grid=x_grid)

_, rRTs_SD = dp.get_standard_individual_rRT_data()
kdes_SD = np.array([gaussian_kde(rRTs)(x_grid) for rRTs in rRTs_SD])
second_peak_locs = np.array([get_secondary_peak_location(kde, x_grid=x_grid) for kde in kdes_SD])
np.sum(~np.isnan(second_peak_locs)*(second_peak_locs < 1.5))/len(second_peak_locs)
import matplotlib.pyplot as plt
import pdb; pdb.set_trace()