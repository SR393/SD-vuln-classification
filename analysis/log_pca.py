import sys
sys.path.extend(['../data'])
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import quantile
from sklearn.decomposition import PCA

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


def plot_geodesic_densities(x, v, alphas, k=0, t_vals=None, ax = None, method = 'kde',show = True):
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

        _, f_plot = pdf_from_quantiles_kde(Q, alphas, x_grid=x_plot, n_samples=20000)

        ax.plot(x_plot, f_plot, label=f"t={t:.3f}")

    ax.set_title(f"Densities along geodesic component {k+1}")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.grid(True)
    ax.legend(ncol=2, fontsize=8)
    if show:
        plt.tight_layout()
        plt.show()

def plot_k_geodesic_densities(x, v, alphas, scores, t_vals = None, method = 'kde'):

    K = v.shape[0]
    if t_vals is None:
        t_vals = np.array([np.linspace(np.min(scores[:,k]), np.max(scores[:,k]), 7) for k in range(K)])

    fig, axs = plt.subplots(1, K)

    for k, ax in enumerate(axs):
        plot_geodesic_densities(x, v[k], alphas=alphas, k=k, t_vals=t_vals[k], ax = ax, method = method, show = False)
    
    plt.tight_layout()
    plt.show()

def plot_k_geodesic_densities_log(x, v, alphas, scores, t_vals = None, method = 'kde'):

    K = v.shape[0]
    if t_vals is None:
        t_vals = np.array([np.linspace(np.min(scores[:,k]), np.max(scores[:,k]), 7) for k in range(K)])

    t_vals, Qs = geodesic_quantiles_from_component(x, v, t_vals=t_vals)

    fig, axs = plt.subplots(1, K)

    for k, ax in enumerate(axs):
        plot_geodesic_densities(x, v[k], alphas=alphas, k=k, t_vals=t_vals[k], ax = ax, method = method, show = False)
    
    plt.tight_layout()
    plt.show()

def log_pca_quantiles(Q, alpha, n_components=2):
    """
    Q: (N, M) array of quantiles Q_i(alpha_m)
    alpha: (M,) grid in (0,1)
    returns: Qbar (M,), components (K,M), scores (N,K), pca object
    """
    Q = np.asarray(Q)
    alpha = np.asarray(alpha)

    # barycenter in 1D-W2 quantile coords
    Qbar = Q.mean(axis=0)

    # centered "log maps"
    W = Q - Qbar[None, :]

    # L2([0,1]) inner product via quadrature weights
    sqrtw = np.sqrt(alpha[1] - alpha[0])

    # weighted PCA = PCA on W * sqrt(w); so vector inner products approximate L2 integrals
    X = W * sqrtw
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)

    # bring components back to quantile-function coordinates
    comps_weighted = pca.components_              # (K, M) in weighted coords
    components = comps_weighted / sqrtw # (K, M) in quantile coords

    return Qbar, components, scores, pca

def compute_quantiles_from_samples(samples_list, M=200, eps=1e-3, method="linear"):
    """
    samples_list: list of 1D arrays, length N
    returns:
      alphas: (M,) in (0,1)
      Q: (N, M) quantile arrays
    """
    alphas = np.linspace(eps, 1 - eps, M)
    Q = np.vstack([np.quantile(s, alphas, method=method) for s in samples_list])
    return alphas, Q

def find_most_similar(scores, k):

    norms = np.linalg.norm(scores, axis=1)
    cos_sims = scores[:, k]/norms
    cos_sim_quants = quantile(cos_sims, [0.01, 0.99])
    high_inds = np.nonzero(cos_sims >= cos_sim_quants[1])[0]
    low_inds = np.nonzero(cos_sims <= cos_sim_quants[0])[0]

    return low_inds, high_inds

def find_extreme_representatives(scores, K = 3):

    '''Find scores and individual indices of individuals who are most similar to extreme ends of principal components.'''

    scores = scores[:, :K]
    rep_indices = np.zeros((K, 2), dtype=int)
    rep_scores = np.zeros((K, 2, K))

    for k in range(K):

        low_inds, high_inds = find_most_similar(scores, k)
        max_ind = high_inds[np.argmax(scores[high_inds, k])]
        min_ind = low_inds[np.argmin(scores[low_inds, k])]
        rep_indices[k, 0], rep_indices[k, 1] = min_ind, max_ind
        rep_scores[k, 0, :], rep_scores[k, 1, :] = scores[min_ind, :], scores[max_ind, :]
        # import pdb; pdb.set_trace()

    return rep_indices, rep_scores

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
# rRTs_1 = [np.log(rRTs_dict_1[ind]) for ind in ind_keys]
# rRTs_2 = [np.log(rRTs_dict_2[ind]) for ind in ind_keys]

q = np.linspace(0, 1.0, 200)
dq = q[1] - q[0]
inds = np.arange(len(ind_keys))
# import pdb; pdb.set_trace()
ind_quants_BL = np.array([quantile(rRTs_1[i], q) for i in inds])
ind_quants_SD = np.array([quantile(rRTs_2[i], q) for i in inds])

K = 8
# t0s = np.linspace(-0.9, 0.9, 3)
t0s = [0]
outputs = []
alphas_1, Q_1 = compute_quantiles_from_samples(rRTs_1, M=200, eps=1e-3, method="linear")
alphas_1, Q_2 = compute_quantiles_from_samples(rRTs_2, M=200, eps=1e-3, method="linear")
# alphas_1 = np.concatenate([np.array([0.0]), alphas_1])
# import pdb; pdb.set_trace()
# Q_1 = np.hstack([np.zeros((Q_1.shape[0], 1)), Q_1])
# Q_2 = np.hstack([np.zeros((Q_2.shape[0], 1)), Q_2])

Qbar_1, components_1, scores_1, pca_1 = log_pca_quantiles(Q_1, alphas_1, n_components=K)
Qbar_2, components_2, scores_2, pca_2 = log_pca_quantiles(Q_2, alphas_1, n_components=K)

np.save(f'../data/logpca_frechet_mean_BL_{K}-dims.npy', Qbar_1)
np.save(f'../data/logpca_components_BL_{K}-dims.npy', components_1)
np.save(f'../data/logpca_coords_BL_{K}-dims.npy', scores_1)
np.save(f'../data/logpca_explained_variance_BL_{K}-dims.npy', pca_1.explained_variance_)
np.save(f'../data/logpca_explained_variance_ratio_BL_{K}-dims.npy', pca_1.explained_variance_ratio_)

np.save(f'../data/logpca_frechet_mean_SD_{K}-dims.npy', Qbar_2)
np.save(f'../data/logpca_components_SD_{K}-dims.npy', components_2)
np.save(f'../data/logpca_coords_SD_{K}-dims.npy', scores_2)
np.save(f'../data/logpca_explained_variance_SD_{K}-dims.npy', pca_2.explained_variance_)
np.save(f'../data/logpca_explained_variance_ratio_SD_{K}-dims.npy', pca_2.explained_variance_ratio_)

r_err_1 = np.zeros(K)
r_err_2 = np.zeros(K)
for k in range(K):
    # import pdb; pdb.set_trace()
    recon_1 = Qbar_1 + np.matmul(scores_1[:, :k+1], components_1[:k+1])
    recon_2 = Qbar_2 + np.matmul(scores_2[:, :k+1], components_2[:k+1])
    r_err_1[k] = np.mean((recon_1 - Q_1)**2)
    r_err_2[k] = np.mean((recon_2 - Q_2)**2)

np.save(f'../data/logpca_recon_error_BL_{K}-dims.npy', r_err_1)
np.save(f'../data/logpca_recon_error_SD_{K}-dims.npy', r_err_2)

pq_grid = np.meshgrid(alphas_1, alphas_1)
pq_grid_arr = np.transpose(np.array(pq_grid), (1, 2, 0))
cov = (np.min(pq_grid_arr, axis=2) - pq_grid_arr[:, :, 0] * pq_grid_arr[:, :, 1])*np.sqrt(alphas_1[1] - alphas_1[0])

noise_est_1 = np.zeros((len(alphas_1), len(alphas_1)))
for i, rRTs in enumerate(rRTs_1):
    kde_dens = gaussian_kde(rRTs)(Q_1[i])
    diag_recip_dens = np.diag(1 / kde_dens)
    noise_i = diag_recip_dens @ cov @ diag_recip_dens/len(rRTs)
    noise_est_1 += noise_i

# import pdb; pdb.set_trace()

noise_est_1 /= len(rRTs_1)
noise_eigvals_1, noise_eigvec_1 = np.linalg.eig(noise_est_1)
plt.plot(np.sort(noise_eigvals_1)[::-1])
plt.plot(pca_1.explained_variance_)
plt.show()

noise_est_2 = np.zeros((len(alphas_1), len(alphas_1)))
for i, rRTs in enumerate(rRTs_2):
    kde_dens = gaussian_kde(rRTs)(Q_2[i])
    diag_recip_dens = np.diag(1 / kde_dens)
    noise_i = diag_recip_dens @ cov @ diag_recip_dens/len(rRTs)
    noise_est_2 += noise_i

noise_est_2 /= len(rRTs_2)
noise_eigvals_2, noise_eigvec_2 = np.linalg.eig(noise_est_2)
plt.plot(np.sort(noise_eigvals_2)[::-1])
plt.plot(pca_2.explained_variance_)
plt.show()

import pdb; pdb.set_trace()

t_vals = np.linspace(-1, 1, 5)
first_comp_geodesics_SD = geodesic_quantiles_from_component(Qbar_2, components_2[0], t_vals=t_vals)
second_comp_geodesics_SD = geodesic_quantiles_from_component(Qbar_2, components_2[1], t_vals=t_vals)
third_comp_geodesics_SD = geodesic_quantiles_from_component(Qbar_2, components_2[2], t_vals=t_vals)

K_to_analyse = 3

rep_indices, rep_scores = find_extreme_representatives(scores_2, K = K_to_analyse)
rep_scores_list = np.reshape(rep_scores, (rep_scores.shape[0]*rep_scores.shape[1], K_to_analyse))
# import pdb; pdb.set_trace()
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(scores_2[:, 0], scores_2[:, 1], scores_2[:, 2], alpha = 0.4, s = 8)
ax.scatter(rep_scores_list[:, 0], rep_scores_list[:, 1], rep_scores_list[:, 2], c = 'r', s = 12)
plt.show()

x_plot = np.linspace(0.01, 6.0, 1000)
# x_plot_log = np.log(x_plot)
# x_plot = np.linspace(np.log(0.01), np.log(6.0), 1000)
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for k in range(K_to_analyse):
    ax = axs[k]
    # import pdb; pdb.set_trace()
    min_ind = rep_indices[k, 0]
    max_ind = rep_indices[k, 1]
    pdf_min = pdf_from_quantiles_kde(Qbar_2 + scores_2[min_ind]@components_2, alphas_1, x_grid = x_plot, n_samples = 20000)[1]
    pdf_max = pdf_from_quantiles_kde(Qbar_2 + scores_2[max_ind]@components_2, alphas_1, x_grid = x_plot, n_samples = 20000)[1]
    # x_plot_n, pdf_min = pdf_of_exp_variable(x_plot, pdf_min)
    # x_plot_n, pdf_max = pdf_of_exp_variable(x_plot, pdf_max)
    # x_plot, pdf_min = log_pdf_to_pdf(x_plot_log, pdf_min)
    # x_plot, pdf_max = log_pdf_to_pdf(x_plot_log, pdf_max)
    ax.plot(x_plot, pdf_min, label = f'Min score ind')
    ax.plot(x_plot, pdf_max, label = f'Max score ind')
    if k == 0:
        ax.legend()
plt.show()



K_to_plot = 3
tvals_1 = np.array([np.linspace(-2*np.std(scores_1[:,k]), 2*np.std(scores_1[:,k]), 7) for k in range(K_to_plot)])
tvals_2 = np.array([np.linspace(-2*np.std(scores_2[:,k]), 2*np.std(scores_2[:,k]), 7) for k in range(K_to_plot)])
import pdb; pdb.set_trace()
plot_k_geodesic_densities(Qbar_1, components_1[:K_to_plot], alphas_1, scores_1[:, :K_to_plot], t_vals = tvals_1, method = 'derivative')
plot_k_geodesic_densities(Qbar_2, components_2[:K_to_plot], alphas_1, scores_2[:, :K_to_plot], t_vals = tvals_2, method = 'derivative')
# import pdb; pdb.set_trace()
# Qhat_1 = Qbar_1[None, :] + scores_1 @ components_1
# Qhat_2 = Qbar_2[None, :] + scores_2 @ components_2

plot_geodesic_densities(Qbar_1, components_1, alphas_1, k=0, t_vals=np.linspace(-1, 1, 7))
plot_geodesic_densities(Qbar_1, components_1, alphas_1, k=1, t_vals=np.linspace(-0.2, 0.3, 7))
plot_geodesic_densities(Qbar_1, components_1, alphas_1, k=2, t_vals=np.linspace(-0.1, 0.1, 7))

plot_geodesic_densities(Qbar_2, components_2, alphas_1, k=0, t_vals=np.linspace(-1, 1, 7))
plot_geodesic_densities(Qbar_2, components_2, alphas_1, k=1, t_vals=np.linspace(-0.4, 0.2, 7))
plot_geodesic_densities(Qbar_2, components_2, alphas_1, k=2, t_vals=np.linspace(-0.1, 0.1, 7))

import pdb; pdb.set_trace()