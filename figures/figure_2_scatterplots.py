import sys
sys.path.extend(['../data', '../analysis'])
import os

import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.cm as mcm

import data_processing_funcs as dp
from log_pca_analysis_funcs import pdf_from_quantiles_kde

def mosaic_scatterplot(beta, hists, edges, axs_list):

    color = 'k'
    beta_1, beta_2 = beta[:, 0], beta[:, 1]
    hist_1, hist_2 = hists[0], hists[1]
    edges_1, edges_2 = edges[0], edges[1]
    axs_scatter, axs_comp_1_hist, axs_comp_2_hist = axs_list

    top_right = np.nonzero((beta_1 > 0)*(beta_2 > 0))[0]
    top_left = np.nonzero((beta_1 < 0)*(beta_2 > 0))[0]
    bottom_right = np.nonzero((beta_1 > 0)*(beta_2 < 0))[0]
    bottom_left = np.nonzero((beta_1 < 0)*(beta_2 < 0))[0]

    axs_scatter.scatter(beta_1[top_right], beta_2[top_right], color=color)
    axs_scatter.scatter(beta_1[top_left], beta_2[top_left], color=color)
    axs_scatter.scatter(beta_1[bottom_right], beta_2[bottom_right], color=color)
    axs_scatter.scatter(beta_1[bottom_left], beta_2[bottom_left], color=color)
    # axs_scatter.vlines(0, ymin=axs_scatter.get_ylim()[0], ymax=axs_scatter.get_ylim()[1], colors='black', linestyles='dashed')
    # axs_scatter.hlines(0, xmin=axs_scatter.get_xlim()[0], xmax=axs_scatter.get_xlim()[1], colors='black', linestyles='dashed')
    # axs_scatter.spines['top'].set_visible(False)
    # axs_scatter.spines['right'].set_visible(False)
    # axs_scatter.spines['left'].set_visible(False)
    # axs_scatter.spines['bottom'].set_visible(False)
    # axs_scatter.yaxis.set_label_position('right') 

    axs_scatter.set_xticks([-1.2, -0.6, 0.0, 0.6, 1.2])
    axs_scatter.set_yticks([-0.4, -0.2, 0.0, 0.2, 0.4])
    axs_scatter.set_xlim(-1.5, 1.5)
    axs_scatter.set_ylim(-0.5, 0.5)
    axs_scatter.set_xlabel('Component 1 Coordinate', size=35, labelpad = 10)
    axs_scatter.set_ylabel('Component 2 Coordinate', size=35, labelpad = 15) 
    axs_scatter.tick_params(axis='both', which='major', labelsize=30)

    axs_comp_1_hist.stairs(hist_1, edges_1, fill = True, color='k', alpha = 0.7)
    axs_comp_1_hist.set_xticks([-1.4, -0.7, 0.0, 0.7, 1.4], labels =['', '', '', '', ''])
    # axs_comp_1_hist.set_xticks([])
    # axs_comp_1_hist.set_yticks([0.0, 0.5, 1.0])
    axs_comp_1_hist.set_yticks([0.2, 0.6, 1.0])
    axs_comp_1_hist.tick_params(axis = 'x', which='major', length = 10)
    axs_comp_1_hist.tick_params(axis='both', which='major', labelsize=30)
    axs_comp_1_hist.set_xlim(-1.5, 1.5)
    axs_comp_1_hist.set_ylim(0.0, 1.2)
    axs_comp_1_hist.set_ylabel('Density', size=30, labelpad = 12.5)
    # axs_comp_1_hist.spines['top'].set_visible(False)
    # axs_comp_1_hist.spines['right'].set_visible(False)
    # axs_comp_1_hist.set_xlabel('Component 1 Coordinate', size=25, labelpad = 20)

    axs_comp_2_hist.stairs(hist_2, edges_2, color='k', fill = True, orientation='horizontal', alpha = 0.7)
    axs_comp_2_hist.set_yticks([-0.4, -0.2, 0.0, 0.2, 0.4], labels =['', '', '', '', ''])
    # axs_comp_2_hist.set_yticks([])
    # axs_comp_2_hist.set_xticks([0.0, 2.0, 4.0])
    hist_2_xticks = [0.5, 2.5, 4.5]
    axs_comp_2_hist.set_xticks(hist_2_xticks, labels = [str(tick) for tick in hist_2_xticks])
    axs_comp_2_hist.tick_params(axis = 'y', which='major', length = 10)
    axs_comp_2_hist.tick_params(axis='both', which='major', labelsize=30)
    axs_comp_2_hist.set_xlim(0.0, 5.0)
    axs_comp_2_hist.set_ylim(-0.5, 0.5)
    # axs_comp_2_hist.spines['top'].set_visible(False)
    # axs_comp_2_hist.spines['right'].set_visible(False)
    # axs_comp_2_hist.set_ylabel('Component 2 Coordinate', size=25, rotation = 270, labelpad = 30)

    # axs_scatter.set_position((axs_scatter.get_position().x0, axs_scatter.get_position().y0, 
    #                           axs_scatter.get_position().width - 0.1, axs_scatter.get_position().height - 0.1))
    # axs_comp_1_hist.set_position((axs_comp_1_hist.get_position().x0 , axs_comp_1_hist.get_position().y0 - 0.01, 
    #                               axs_comp_1_hist.get_position().width - 0.1, axs_comp_1_hist.get_position().height))
    # axs_comp_2_hist.set_position((axs_comp_2_hist.get_position().x0 - 0.05, axs_comp_2_hist.get_position().y0, 
    #                               axs_comp_2_hist.get_position().width, axs_comp_2_hist.get_position().height - 0.1))

    return {'scatter': axs_scatter, 'comp_1_hist': axs_comp_1_hist, 'comp_2_hist': axs_comp_2_hist}

def insert_density(ax, x, density, color):

    ax.plot(x, density, color=color, linewidth=2)
    ax.fill_between(x, density, color=color, alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    return ax

def shift_submosaic(axs, x_shift = 0.0, y_shift = 0.0):

    for ax in axs:
        ax.set_position((ax.get_position().x0 + x_shift, ax.get_position().y0 + y_shift, 
                         ax.get_position().width, ax.get_position().height) )

max_K = 8
M = 200
eps = 1e-3
alphas = np.linspace(eps, 1 - eps, M)

log_Qbar_BL = np.load(f'../data/logpca_frechet_mean_BL_{max_K}-dims.npy')
components_BL = np.load(f'../data/logpca_components_BL_{max_K}-dims.npy')
beta_BL = np.load(f'../data/logpca_coords_BL_{max_K}-dims.npy')
beta_BL[:, 1] = -beta_BL[:, 1]  # reflect component 2 so negative values indicate poorer performance
components_BL[1] = -components_BL[1]

log_Qbar_SD = np.load(f'../data/logpca_frechet_mean_SD_{max_K}-dims.npy')
components_SD = np.load(f'../data/logpca_components_SD_{max_K}-dims.npy')
beta_SD = np.load(f'../data/logpca_coords_SD_{max_K}-dims.npy')

hist_1_BL, edges_1_BL = np.histogram(beta_BL[:, 0], bins=15, density=True)
hist_1_SD, edges_1_SD = np.histogram(beta_SD[:, 0], bins=15, density=True)
hist_2_BL, edges_2_BL = np.histogram(beta_BL[:, 1], bins=15, density=True)
hist_2_SD, edges_2_SD = np.histogram(beta_SD[:, 1], bins=15, density=True)
hists_BL, edges_BL = [hist_1_BL, hist_2_BL], [edges_1_BL, edges_2_BL]
hists_SD, edges_SD = [hist_1_SD, hist_2_SD], [edges_1_SD, edges_2_SD]

x_grid = np.linspace(-1.0, 6.5, 1000)

labelsize = 50

fig, axs = plt.subplot_mosaic([['comp_1_hist_BL', '.', '.', 'comp_1_hist_SD', '.'],
                               ['scatter_BL', 'comp_2_hist_BL', '.', 'scatter_SD', 'comp_2_hist_SD']],
                               figsize=(28, 14),
                               width_ratios=(4, 1, 0.5, 4, 1), height_ratios=(1, 4),
                               layout='constrained')

### Baseline ###

# colours_BL = ["#0aa70a", '#0000ff', "#08b3b3", "#920392"]

quant_top_right = log_Qbar_BL + 0.8*components_BL[0] + 0.2*components_BL[1]
quant_top_left = log_Qbar_BL - 0.8*components_BL[0] + 0.2*components_BL[1]
quant_bottom_right = log_Qbar_BL + 0.8*components_BL[0] - 0.2*components_BL[1]
quant_bottom_left = log_Qbar_BL - 0.8*components_BL[0] - 0.2*components_BL[1]

dens_top_right = pdf_from_quantiles_kde(quant_top_right, alphas, x_grid=x_grid, eps=1e-3)[1]
dens_top_left = pdf_from_quantiles_kde(quant_top_left, alphas, x_grid=x_grid, eps=1e-3)[1]
dens_bottom_right = pdf_from_quantiles_kde(quant_bottom_right, alphas, x_grid=x_grid, eps=1e-3)[1]
dens_bottom_left = pdf_from_quantiles_kde(quant_bottom_left, alphas, x_grid=x_grid, eps=1e-3)[1]

axs['scatter_BL'].set_xlim(-1.5, 1.5)
axs['scatter_BL'].set_ylim(-0.5, 0.5)
axs['comp_1_hist_BL'].set_xlim(-1.5, 1.5)
axs['comp_2_hist_BL'].set_ylim(-0.5, 0.5)

axs['comp_1_hist_BL'].text(-0.15, 1.2, 'A', fontsize=labelsize, transform=axs['comp_1_hist_BL'].transAxes)
axs['comp_1_hist_BL'].text(0.5, 1.2, 'Baseline', ha='center', fontsize=labelsize, transform=axs['comp_1_hist_BL'].transAxes)

mosaic_scatterplot(beta_BL, hists_BL, edges_BL, 
                #    colours_BL,
                   [axs['scatter_BL'], axs['comp_1_hist_BL'], axs['comp_2_hist_BL']])

dens_ax_top_right = axs['scatter_BL'].inset_axes([0.7, 0.7, 0.25, 0.25])
dens_ax_top_left = axs['scatter_BL'].inset_axes([0.05, 0.7, 0.25, 0.25])
dens_ax_bottom_right = axs['scatter_BL'].inset_axes([0.7, 0.05, 0.25, 0.25])
dens_ax_bottom_left = axs['scatter_BL'].inset_axes([0.05, 0.05, 0.25, 0.25])

dens_col = 'blue'
dens_ax_top_right = insert_density(dens_ax_top_right, x_grid, dens_top_right, dens_col)
dens_ax_top_left = insert_density(dens_ax_top_left, x_grid, dens_top_left, dens_col)
dens_ax_bottom_right = insert_density(dens_ax_bottom_right, x_grid, dens_bottom_right, dens_col)
dens_ax_bottom_left = insert_density(dens_ax_bottom_left, x_grid, dens_bottom_left, dens_col)
# dens_ax_top_right = insert_density(dens_ax_top_right, x_grid, dens_top_right, colours_BL[0])
# dens_ax_top_left = insert_density(dens_ax_top_left, x_grid, dens_top_left, colours_BL[1])
# dens_ax_bottom_right = insert_density(dens_ax_bottom_right, x_grid, dens_bottom_right, colours_BL[2])
# dens_ax_bottom_left = insert_density(dens_ax_bottom_left, x_grid, dens_bottom_left, colours_BL[3])

dens_ax_top_right.set_xlabel('rRTs', size=30)
dens_ax_top_right.set_ylabel('Density', size=30)


### SD ###

colours_SD = ["#6dff0b", "#ff5900", "#daa520", "#c30000"]

x_grid = np.linspace(-1.0, 6.5, 1000)

quant_top_right = log_Qbar_SD + 0.8*components_SD[0] + 0.2*components_SD[1]
quant_top_left = log_Qbar_SD - 0.8*components_SD[0] + 0.2*components_SD[1]
quant_bottom_right = log_Qbar_SD + 0.8*components_SD[0] - 0.2*components_SD[1]
quant_bottom_left = log_Qbar_SD - 0.8*components_SD[0] - 0.2*components_SD[1]

dens_top_right = pdf_from_quantiles_kde(quant_top_right, alphas, x_grid=x_grid, eps=1e-3)[1]
dens_top_left = pdf_from_quantiles_kde(quant_top_left, alphas, x_grid=x_grid, eps=1e-3)[1]
dens_bottom_right = pdf_from_quantiles_kde(quant_bottom_right, alphas, x_grid=x_grid, eps=1e-3)[1]
dens_bottom_left = pdf_from_quantiles_kde(quant_bottom_left, alphas, x_grid=x_grid, eps=1e-3)[1]

axs['scatter_SD'].set_xlim(-1.5, 1.5)
axs['scatter_SD'].set_ylim(-0.5, 0.5)
axs['comp_1_hist_SD'].set_xlim(-1.5, 1.5)
axs['comp_2_hist_SD'].set_ylim(-0.5, 0.5)

axs['comp_1_hist_SD'].text(-0.15, 1.2, 'B', fontsize=labelsize, transform=axs['comp_1_hist_SD'].transAxes)
axs['comp_1_hist_SD'].text(0.5, 1.2, 'Sleep-Deprived', ha = 'center', fontsize=labelsize, transform=axs['comp_1_hist_SD'].transAxes)

mosaic_scatterplot(beta_SD, hists_SD, edges_SD, 
                  # colours_SD,
                  [axs['scatter_SD'], axs['comp_1_hist_SD'], axs['comp_2_hist_SD']])

dens_ax_top_right = axs['scatter_SD'].inset_axes([0.7, 0.7, 0.25, 0.25])
dens_ax_top_left = axs['scatter_SD'].inset_axes([0.05, 0.7, 0.25, 0.25])
dens_ax_bottom_right = axs['scatter_SD'].inset_axes([0.7, 0.05, 0.25, 0.25])
dens_ax_bottom_left = axs['scatter_SD'].inset_axes([0.05, 0.05, 0.25, 0.25])

dens_col = 'blue'
dens_ax_top_right = insert_density(dens_ax_top_right, x_grid, dens_top_right, dens_col)
dens_ax_top_left = insert_density(dens_ax_top_left, x_grid, dens_top_left, dens_col)
dens_ax_bottom_right = insert_density(dens_ax_bottom_right, x_grid, dens_bottom_right, dens_col)
dens_ax_bottom_left = insert_density(dens_ax_bottom_left, x_grid, dens_bottom_left, dens_col)
# dens_ax_top_right = insert_density(dens_ax_top_right, x_grid, dens_top_right, colours_SD[0])
# dens_ax_top_left = insert_density(dens_ax_top_left, x_grid, dens_top_left, colours_SD[1])
# dens_ax_bottom_right = insert_density(dens_ax_bottom_right, x_grid, dens_bottom_right, colours_SD[2])
# dens_ax_bottom_left = insert_density(dens_ax_bottom_left, x_grid, dens_bottom_left, colours_SD[3])

plt.savefig(f'figure_2_coordinate_scatterplots.pdf', dpi = 600)

