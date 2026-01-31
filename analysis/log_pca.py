import sys
sys.path.extend(['../data'])

import numpy as np
from sklearn.decomposition import PCA

import data_processing_funcs as dp
from log_pca_analysis_funcs import compute_quantiles_from_samples

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

rRTs_1, rRTs_2 = dp.get_standard_individual_rRT_data()

K = 8

alphas_1, Q_1 = compute_quantiles_from_samples(rRTs_1, M=200, eps=1e-3, method="linear")
alphas_1, Q_2 = compute_quantiles_from_samples(rRTs_2, M=200, eps=1e-3, method="linear")

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



