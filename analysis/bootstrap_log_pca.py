import sys
sys.path.extend(['../data'])
import os
from tqdm import tqdm

import numpy as np

import data_processing_funcs as dp
from log_pca_analysis_funcs import compute_quantiles_from_samples, log_pca_quantiles

files = os.listdir('../data/bootstrap_samples/RTs/')
K = 8

if not os.path.exists('../data/bootstrap_samples/q_means'):
    os.makedirs('../data/bootstrap_samples/q_means')
if not os.path.exists('../data/bootstrap_samples/components'):
    os.makedirs('../data/bootstrap_samples/components')
if not os.path.exists('../data/bootstrap_samples/coordinates'):
    os.makedirs('../data/bootstrap_samples/coordinates')
if not os.path.exists('../data/bootstrap_samples/explained_variance'):
    os.makedirs('../data/bootstrap_samples/explained_variance')
if not os.path.exists('../data/bootstrap_samples/explained_variance_ratio'):
    os.makedirs('../data/bootstrap_samples/explained_variance_ratio')

for f in tqdm(files, total = len(files)):
    RTs_dict = dp.json_to_dict(f'../data/bootstrap_samples/RTs/{f}')
    index = RTs_dict['bs_index']
    del RTs_dict['bs_index']
    RTs_dict = {pid: [np.array(session) for session in sessions] for pid, sessions in RTs_dict.items()}
    rRTs_1, rRTs_2 = dp.get_standard_individual_rRT_data(RTs_dict)

    alphas_1, Q_1 = compute_quantiles_from_samples(rRTs_1, M=200, eps=1e-3, method="linear")
    alphas_2, Q_2 = compute_quantiles_from_samples(rRTs_2, M=200, eps=1e-3, method="linear")

    Qbar_1, components_1, scores_1, pca_1 = log_pca_quantiles(Q_1, alphas_1, n_components=K)
    Qbar_2, components_2, scores_2, pca_2 = log_pca_quantiles(Q_2, alphas_1, n_components=K)

    np.save(f'../data/bootstrap_samples/q_means/logpca_frechet_mean_BL_{K}-dims_{index}.npy', Qbar_1)
    np.save(f'../data/bootstrap_samples/components/logpca_components_BL_{K}-dims_{index}.npy', components_1)
    np.save(f'../data/bootstrap_samples/coordinates/logpca_coords_BL_{K}-dims_{index}.npy', scores_1)
    np.save(f'../data/bootstrap_samples/explained_variance/logpca_explained_variance_BL_{K}-dims_{index}.npy', pca_1.explained_variance_)
    np.save(f'../data/bootstrap_samples/explained_variance_ratio/logpca_explained_variance_ratio_BL_{K}-dims_{index}.npy', pca_1.explained_variance_ratio_)

    np.save(f'../data/bootstrap_samples/q_means/logpca_frechet_mean_SD_{K}-dims_{index}.npy', Qbar_2)
    np.save(f'../data/bootstrap_samples/components/logpca_components_SD_{K}-dims_{index}.npy', components_2)
    np.save(f'../data/bootstrap_samples/coordinates/logpca_coords_SD_{K}-dims_{index}.npy', scores_2)
    np.save(f'../data/bootstrap_samples/explained_variance/logpca_explained_variance_SD_{K}-dims_{index}.npy', pca_2.explained_variance_)
    np.save(f'../data/bootstrap_samples/explained_variance_ratio/logpca_explained_variance_ratio_SD_{K}-dims_{index}.npy', pca_2.explained_variance_ratio_)