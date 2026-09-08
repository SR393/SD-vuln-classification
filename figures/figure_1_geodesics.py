import sys
sys.path.extend(['../data', '../analysis'])

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.cm as mcm
from matplotlib import colormaps
import seaborn as sns

import data_processing_funcs as dp
from log_pca_analysis_funcs import geodesic_quantiles_from_component, pdf_from_quantiles_kde

def plot_geodesic_densities(fig = None, ax = None, k = 1, cond_str = '', dens_args = None, cbar_args = None):

    Qbar, v, alphas, t_vals = dens_args
    cmap, cbar_pos, cbar_ticks = cbar_args

    t_vals, Qs = geodesic_quantiles_from_component(Qbar, v, t_vals=t_vals)
    x_plot = np.linspace(-1.0, 6.0, 400)

    norm = col.Normalize(vmin=np.min(t_vals), vmax=np.max(t_vals))
    cmap = colormaps[cmap]
    
    if ax is None:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111)

    add_inset_colorbar(fig, ax, cmap, cbar_pos, cbar_ticks, norm, label=rf"$\beta^{{{cond_str}}}_{k}$")

    for t, Q in zip(t_vals, Qs):

        _, f_plot = pdf_from_quantiles_kde(Q, alphas, x_grid=x_plot)
        # import pdb; pdb.set_trace()
        ax.plot(x_plot, f_plot, # label=rf'$\beta^{{{cond_str}}}_{k}$ = ' + f"{t:.3f}", 
                color = colormaps.get_cmap(cmap)(norm(t)), lw = 1.5)

    if cond_str == 'SD':
        ax.vlines(0.0, ymin=0.0, ymax=ax.get_ylim()[1], colors='k', linestyles='dashed', linewidth=1)
        ax.fill_betweenx(y=[0.0, ax.get_ylim()[1]], x1=np.min(x_plot), x2=0.0, color='lightgrey', alpha=0.5)

    title = f"{cond_str} Component {k}"

    ax.set_title(title, size = 35, x = 0.075, y = 1.05, ha = 'left', va = 'top', transform=ax.transAxes)
    ax.set_xlabel(r"rRT ($s^{-1}$)", size = 35, labelpad = 12)
    ax.set_ylabel("Density", size = 35, labelpad = 12)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=30)

def add_inset_colorbar(fig, ax, cmap, pos, cbar_ticks, norm, label="Component coordinate"):
    cax = ax.inset_axes(pos)

    sm = mcm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        cax=cax,
        orientation="horizontal"
    )
    
    cbar.set_label(label, fontsize=18, labelpad=6)
    cbar.set_ticks(cbar_ticks)
    cbar.ax.tick_params(labelsize=15)

    return cbar

Qbar_BL = np.load('../data/logpca_frechet_mean_BL_8-dims.npy')
components_BL = np.load('../data/logpca_components_BL_8-dims.npy')
components_BL[1] = -components_BL[1]  # reflect component 2 so negative values indicate poorer performance
explained_variance_BL = np.load('../data/logpca_explained_variance_BL_8-dims.npy')
explained_variance_ratio_BL = np.load('../data/logpca_explained_variance_ratio_BL_8-dims.npy')

Qbar_SD = np.load('../data/logpca_frechet_mean_SD_8-dims.npy')
components_SD = np.load('../data/logpca_components_SD_8-dims.npy')
explained_variance_SD = np.load('../data/logpca_explained_variance_SD_8-dims.npy')
explained_variance_ratio_SD = np.load('../data/logpca_explained_variance_ratio_SD_8-dims.npy')

t_vals_BL = np.array([np.linspace(-2*np.sqrt(explained_variance_BL[k]), 2*np.sqrt(explained_variance_BL[k]), 7) for k in range(8)])
t_vals_SD = np.array([np.linspace(-2*np.sqrt(explained_variance_SD[k]), 2*np.sqrt(explained_variance_SD[k]), 7) for k in range(8)])

eps = 1e-3
M = 200
alphas = np.linspace(eps, 1 - eps, M)

x_plot_BL = np.linspace(0.0, 6.5, 400)
x_plot_SD = np.linspace(-0.75, 6.5, 400)

label_xloc_left = -0.0325
label_xloc_right = -0.0325
label_yloc = 1.0
labelsize = 45

low_xlim = -1.0
high_xlim = 7.0

left_shift = 0.01
right_shift = 0.04
up_shift = 0.04
down_shift = 0.01

# # Create a colormap (hue 250 = blue, hue 15 = red) with a dark center
# seaborn_dark_cmap = sns.diverging_palette(
#     15, 250, center="dark", as_cmap=True
# )

fig = plt.figure(figsize=(12.5, 12))
#elb_plot_ax = fig.add_subplot(221)
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(223)
ax3 = fig.add_subplot(222)
ax4 = fig.add_subplot(224)

cmap = 'cividis'

dens_args_BL_1 = (Qbar_BL, components_BL[0], alphas, t_vals_BL[0])
cbar_args_BL_1 = (cmap, [0.03, 0.85, 0.45, 0.03], [-0.5, 0.0, 0.5])
plot_geodesic_densities(fig = fig, ax = ax1, k = 1, cond_str = 'BL', 
                        dens_args = dens_args_BL_1, cbar_args = cbar_args_BL_1)
ax1.set_xticks([0.0, 2.0, 4.0, 6.0], labels = ['0.0', '2.0', '4.0', '6.0'])
ax1.set_yticks([0.0, 0.4, 0.8])
ax1.set_xlabel('')
ax1.set_ylim(-0.03, 0.95)
ax1.set_xlim(low_xlim, high_xlim)
ax1.text(label_xloc_left, label_yloc, 'A', fontsize=labelsize, 
         ha='right', va='bottom', transform=ax1.transAxes)
ax1.set_position((ax1.get_position().x0 - left_shift, 
                  ax1.get_position().y0 + up_shift,
                  ax1.get_position().x1 - ax1.get_position().x0, 
                  ax1.get_position().y1 - ax1.get_position().y0))
# ax1.legend(ncol=1, fontsize=11.0, loc =(0.02, 0.4))
# plt.savefig('logpca_BL_component1_geodesic_densities.png', dpi = 600)
# plt.close()

# fig = plt.figure(figsize=(12, 12))
# ax = fig.add_subplot(222)
dens_args_BL_2 = (Qbar_BL, components_BL[1], alphas, t_vals_BL[1])
cbar_args_BL_2 = (cmap, [0.03, 0.85, 0.45, 0.03], [-0.15, 0.0, 0.15])
plot_geodesic_densities(fig = fig, ax = ax2, k = 2, cond_str = 'BL', 
                        dens_args = dens_args_BL_2, cbar_args = cbar_args_BL_2)
ax2.set_xlim(low_xlim, high_xlim)
ax2.set_ylim(-0.03, 1.2)
ax2.set_xticks([0.0, 2.0, 4.0, 6.0], labels = ['0.0', '2.0', '4.0', '6.0'])
ax2.set_yticks([0.0, 0.5, 1.0])
ax2.text(label_xloc_left, label_yloc, 'B', fontsize=labelsize, 
         ha='right', va='bottom', transform=ax2.transAxes)
ax2.set_position((ax2.get_position().x0 - left_shift, 
                  ax2.get_position().y0 - down_shift,
                  ax2.get_position().x1 - ax2.get_position().x0, 
                  ax2.get_position().y1 - ax2.get_position().y0))
# ax2.legend(ncol=1, fontsize=11.0, loc =(0.02, 0.4))


dens_args_SD_1 = (Qbar_SD, components_SD[0], alphas, t_vals_SD[0])
cbar_args_SD_1 = (cmap, [0.51, 0.85, 0.45, 0.03], [-0.5, 0.0, 0.5])
plot_geodesic_densities(fig = fig, ax = ax3, k = 1, cond_str = 'SD', 
                        dens_args = dens_args_SD_1, cbar_args = cbar_args_SD_1)
ax3.set_xticks([0.0, 2.0, 4.0, 6.0], labels = ['0.0', '2.0', '4.0', '6.0'])
ax3.set_yticks([0.0, 0.4, 0.8])
ax3.set_xlabel('')
ax3.set_ylabel('')
ax3.set_xlim(low_xlim, high_xlim)
ax3.set_ylim(-0.03,0.95)
ax3.text(label_xloc_right, label_yloc, 'C', fontsize=labelsize, 
         ha='right', va='bottom', transform=ax3.transAxes)
ax3.set_position((ax3.get_position().x0 + right_shift, 
                  ax3.get_position().y0 + up_shift, 
                  ax3.get_position().x1 - ax3.get_position().x0, 
                  ax3.get_position().y1 - ax3.get_position().y0))
# ax3.legend(ncol=1, fontsize=11.0, loc =(0.7, 0.4))

dens_args_SD_2 = (Qbar_SD, components_SD[1], alphas, t_vals_SD[1])
cbar_args_SD_2 = (cmap, [0.51, 0.85, 0.45, 0.03], [-0.2, 0.0, 0.2])
plot_geodesic_densities(fig = fig, ax = ax4, k = 2, cond_str = 'SD', 
                        dens_args = dens_args_SD_2, cbar_args = cbar_args_SD_2)
ax4.set_xlim(low_xlim, high_xlim)
ax4.set_ylim(-0.03, 1.2)
ax4.set_ylabel('')
ax4.set_xticks([0.0, 2.0, 4.0, 6.0], labels = ['0.0', '2.0', '4.0', '6.0'])
ax4.set_yticks([0.0, 0.5, 1.0])
ax4.text(label_xloc_right, label_yloc, 'D', fontsize=labelsize, 
         ha='right', va='bottom', transform=ax4.transAxes)
ax4.set_position((ax4.get_position().x0 + right_shift, 
                  ax4.get_position().y0 - down_shift, 
                  ax4.get_position().x1 - ax4.get_position().x0, 
                  ax4.get_position().y1 - ax4.get_position().y0))
# ax4.legend(ncol=1, fontsize=11.0, loc = (0.7, 0.4))

plt.savefig('figure_1_aug_2026.pdf', dpi = 600)
plt.close()




