import sys
sys.path.extend(['../data', '../fitting'])
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

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

    Qs = []
    for t in t_vals:
        s = t
        Q = x + s * v_k
        Qs.append(Q)
    return t_vals, Qs

def pdf_from_quantiles_kde(Q, alphas, x_grid=None, n_samples=20000, eps=1e-3):
    Q = np.asarray(Q)
    alphas = np.asarray(alphas)

    # sample U in (eps, 1-eps) to avoid tail craziness
    u = np.random.uniform(eps, 1 - eps, size=n_samples)
    # import pdb; pdb.set_trace()
    samples = np.interp(u, alphas, Q)

    if x_grid is None:
        lo, hi = np.quantile(samples, [0.001, 0.999])
        x_grid = np.linspace(lo, hi, 400)

    kde = gaussian_kde(samples)
    return x_grid, kde(x_grid)

def plot_geodesic_densities(x, v, alphas, t_vals=None, ax = None, show = True):
    t_vals, Qs = geodesic_quantiles_from_component(x, v, t_vals=t_vals)

    # Choose a common x-grid for plotting
    # (use middle t as a reference)
    q_mid = Qs[len(Qs)//2]
    # import pdb; pdb.set_trace()
    x_plot = np.linspace(-1.0, 5.6, 400)
    if ax is None:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111)

    for t, Q in zip(t_vals, Qs):
        # if not is_monotone_nondec(Q):
            # continue  # skip invalid steps

        _, f_plot = pdf_from_quantiles_kde(Q, alphas, x_grid=x_plot, n_samples=20000)

        ax.plot(x_plot, f_plot, label=f"t={t:.3f}")

    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.grid(True)
    ax.legend(ncol=2, fontsize=8)
    if show:
        plt.tight_layout()
        plt.show()

max_K = 8

beta_SD = np.load(f'../data/logpca_coords_SD_{max_K}-dims.npy')
log_Qbar_SD = np.load(f'../data/logpca_frechet_mean_SD_{max_K}-dims.npy')
components_SD = np.load(f'../data/logpca_components_SD_{max_K}-dims.npy')
explained_variance_SD = np.load(f'../data/logpca_explained_variance_SD_{max_K}-dims.npy')

ranges = np.array([-2*np.sqrt(explained_variance_SD), 2*np.sqrt(explained_variance_SD)]).T
t_vals = np.array([np.linspace(r[0], r[1], 5) for r in ranges])

comp_1_geo = geodesic_quantiles_from_component(log_Qbar_SD, components_SD[0], t_vals=t_vals[0])[1]
comp_2_geo = geodesic_quantiles_from_component(log_Qbar_SD, components_SD[1], t_vals=t_vals[1])[1]

M = 200
eps = 1e-3
alphas = np.linspace(eps, 1 - eps, M)

# for i, Q_0 in enumerate(comp_1_geo):

#     fig, ax = plt.subplots(1, 1, figsize=(10, 10))
#     plot_geodesic_densities(Q_0, components_SD[2], alphas=alphas, ax=ax, t_vals=t_vals[2], show=False)
#     ax.set_title(f'Comp 3 Geodesic From Comp 1 t={np.round(t_vals[0, i], 2)}', size=20)
#     plt.savefig(f'figures/geodesics/geodesic_densities_comp3_from_comp1_t{np.round(t_vals[0, i], 2)}.png')
#     plt.show()

# for i, Q_0 in enumerate(comp_2_geo):

#     fig, ax = plt.subplots(1, 1, figsize=(10, 10))
#     plot_geodesic_densities(Q_0, components_SD[2], alphas=alphas, ax=ax, t_vals=t_vals[2], show=False)
#     ax.set_title(f'Comp 3 Geodesic From Comp 2 t={np.round(t_vals[1, i], 2)}', size=20)
#     plt.savefig(f'figures/geodesics/geodesic_densities_comp3_from_comp2_t{np.round(t_vals[1, i], 2)}.png')
#     plt.show()

fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(beta_SD[:, 0], beta_SD[:, 1], beta_SD[:, 2], alpha = 0.7, s = 10)
ax.set_xlabel('Log-PCA Comp 1 (Primary Peak)', size = 15)
ax.set_ylabel('Log-PCA Comp 2 (Width)', size = 15)
ax.set_zlabel('Log-PCA Comp 3', size = 15)
ax.plot(ranges[0], np.array([0, 0]), np.array([0, 0]), color = 'k', lw = 1)
ax.plot(np.array([0, 0]), ranges[1], np.array([0, 0]), color = 'k', lw = 1)
ax.plot(np.array([0, 0]), np.array([0, 0]), ranges[2], color = 'k', lw = 1)

for i, t1 in enumerate(t_vals[0]):
    ax.plot([t1, t1], [0, 0], [ranges[2, 0], ranges[2, -1]], color = 'r', lw = 0.8, ls = '--')
for i, t2 in enumerate(t_vals[1]):
    ax.plot([0, 0], [t2, t2], [ranges[2, 0], ranges[2, -1]], color = 'g', lw = 0.8, ls = '--')
import pdb; pdb.set_trace()
ax.grid(False)
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])
plt.show()