import sys
sys.path.extend(['../data', '../fitting'])
import os

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.colors as col
from matplotlib import colormaps
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_samples
import seaborn as sns
from scipy.stats import quantile

import data_processing_funcs as dp
from characterisation_functions import transform_pars, calculate_am, get_second_peak_indices
from RM_fitting_functions import pdf

def get_transition_matrix(inds_1, inds_2):

    """Create a transition matrix between two sets of indices."""
    n_1 = len(set(inds_1))
    n_2 = len(set(inds_2))
    transition_matrix = np.zeros((n_1, n_2))
    
    for i in range(n_1):
        for j in range(n_2):
            transition_matrix[i, j] = np.sum((inds_1 == i) & (inds_2 == j))/np.sum(inds_1 == i)
    
    return transition_matrix

def plot_transition_matrix(inds_1, inds_2, ax = None):

    """Plot the transition matrix between two sets of indices."""
    transition_matrix = get_transition_matrix(inds_1, inds_2)

    n_1 = len(set(inds_1))
    n_2 = len(set(inds_2))

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(transition_matrix, annot=True, cmap='viridis', linewidths=0.5, ax=ax)
    
    ax.set_xticks(np.arange(n_2) + 0.5)
    ax.set_yticks(np.arange(n_1) + 0.5)
    
    ax.set_xticklabels([f'SD Group {i}' for i in range(n_2)])
    ax.set_yticklabels([f'BL Group {i}' for i in range(n_1)])
    
    ax.set_xlabel('SD Groups')
    ax.set_ylabel('BL Groups')
    ax.set_title('Transition Matrix')

def plot_clusters(points, group_assign, param_indices, parameter_names, silhouettes,title, ax = None):
    """Plot clusters in 3D space."""
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
    cmap = plt.get_cmap('tab10' if max(group_assign) < 10 else 'tab20')
    
    for group in np.unique(group_assign):
        mask = group_assign == group
        # import pdb; pdb.set_trace()
        col = cmap((np.ones(sum(mask))*group).astype(int))
        col[:, 3] = (silhouettes[mask] + 1)/2
        ax.scatter(
            points[mask, param_indices[0]],
            points[mask, param_indices[1]],
            points[mask, param_indices[2]],
            label=f'Group {group}',
            color=col,
            # alpha=(silhouettes[mask] + 1)/2,
            s=5
        )
    
    ax.set_xlabel(parameter_names[param_indices[0]])
    ax.set_ylabel(parameter_names[param_indices[1]])
    ax.set_zlabel(parameter_names[param_indices[2]])
    ax.legend()
    ax.set_title(title)

def plot_silhouettes(points, silhouettes, param_indices, parameter_names, title, ax = None):
    """Plot clusters in 3D space."""
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
    cmap = plt.get_cmap('coolwarm')
    norm = col.Normalize(vmin=-1, vmax=1)
    
    ax.scatter(
        points[:, param_indices[0]],
        points[:, param_indices[1]],
        points[:, param_indices[2]],
        color=cmap(norm(silhouettes)),
        alpha=0.8,
        s=5
    )
    
    ax.set_xlabel(parameter_names[param_indices[0]])
    ax.set_ylabel(parameter_names[param_indices[1]])
    ax.set_zlabel(parameter_names[param_indices[2]])
    ax.set_title(title)

def wass_p_dist(quant_1, quant_2, dq, p = 2):

    """Calculate the Wasserstein p distance over x between two cdfs."""
    dist = np.sum(np.abs(quant_1 - quant_2)**p*dq)**(1/p)
    return dist

def calc_dist_mat(rRTs, p):
    """Calculate the affinity matrix for clustering."""

    n_points = len(rRTs)
    affinity_matrix = np.zeros((n_points, n_points))
    q = np.linspace(0.0, 1.0, 10000)
    dq = q[1] - q[0]

    for i in range(n_points):
        quantile_i = quantile(rRTs[i], q)
        for j in range(i + 1, n_points):
            quantile_j = quantile(rRTs[j], q)
            dist = wass_p_dist(quantile_i, quantile_j, dq, p = p)
            # dist = wass_dist(cdf_i, cdf_j, dx)
            affinity_matrix[i, j] = dist
            affinity_matrix[j, i] = dist

    return affinity_matrix

def sort_groups_by_ah(points, group_assign):

    """Sort groups by mean a_h value."""

    mean_ah_list = np.array([np.mean(points[group_assign == group, 1]) for group in np.unique(group_assign)])
    sorted_groups = np.argsort(mean_ah_list)
    new_group_assign = np.zeros_like(group_assign)

    for i, group in enumerate(sorted_groups):
        new_group_assign[group_assign == group] = i

    return new_group_assign

# --- Choose which parameters to plot (indices 0-3) ---
param_indices = [1, 2]  # Change as desired
parameter_names = ['s', r'$a_h$', r'$a_l$']

ind_keys, pars_arr = dp.get_ind_fit_data_arr()
theta1 = transform_pars(pars_arr[0])
theta2 = transform_pars(pars_arr[1])

a_m_1 = calculate_am(pars_arr[0])
a_m_2 = calculate_am(pars_arr[1])

x = np.linspace(0.01, 6.25, 10000)

SP_area_1 = np.empty(len(ind_keys))
SP_area_2 = np.empty(len(ind_keys))

for i, (ind_pars_1, ind_pars_2) in enumerate(zip(pars_arr[0], pars_arr[1])):

    pdf_1 = pdf(ind_pars_1, x)
    pdf_2 = pdf(ind_pars_2, x)
    trough_index_1 = np.argmin(np.abs(x - a_m_1[i]))
    trough_index_2 = np.argmin(np.abs(x - a_m_2[i]))
    SP_area_1[i] = np.trapz(pdf_1[:trough_index_1], x[:trough_index_1])
    SP_area_2[i] = np.trapz(pdf_2[:trough_index_2], x[:trough_index_2])

BL_points = np.array([theta1[:, 0], theta1[:, 1], theta1[:, 2], SP_area_1]).T
SD_points = np.array([theta2[:, 0], theta2[:, 1], theta2[:, 2], SP_area_2]).T

session_ind_arr_1 = np.arange(8)
session_ind_arr_2 = np.arange(8, 20)

RTs_dict = dp.load_RT_data_dict(raw = False, subs = None)
RTs_dict = dp.filter_data(RTs_dict, lower_cutoff = 167, upper_cutoff = 10000, 
                            remove_circ_misaligned = True)
rRTs_dict_1 = dp.get_individual_data(RTs_dict, session_ind_arr_1, scale_factor = 1000)
rRTs_dict_2 = dp.get_individual_data(RTs_dict, session_ind_arr_2, scale_factor = 1000)
rRTs_1 = [rRTs_dict_1[ind] for ind in ind_keys]
rRTs_2 = [rRTs_dict_2[ind] for ind in ind_keys]

p = 2
dist_func_str = f'emp_wass_{p}'

if not os.path.exists(f'../data/{dist_func_str}_dist_BL.npy'):
    BL_dist_mat = calc_dist_mat(rRTs_1, p = p)
    np.save(f'../data/{dist_func_str}_dist_BL.npy', BL_dist_mat)
else:
    BL_dist_mat = np.load(f'../data/{dist_func_str}_dist_BL.npy')

if not os.path.exists(f'../data/{dist_func_str}_dist_SD.npy'):
    SD_dist_mat = calc_dist_mat(rRTs_2, p = p)
    np.save(f'../data/{dist_func_str}_dist_SD.npy', SD_dist_mat)
else:
    SD_dist_mat = np.load(f'../data/{dist_func_str}_dist_SD.npy')

BL_dists_unique = BL_dist_mat[np.triu_indices(BL_dist_mat.shape[0], k = 1)].flatten()
SD_dists_unique = SD_dist_mat[np.triu_indices(SD_dist_mat.shape[0], k = 1)].flatten()

hist_BL, edges_BL = np.histogram(BL_dists_unique, bins=50)
hist_SD, edges_SD = np.histogram(SD_dists_unique, bins=50)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].stairs(hist_BL, edges_BL)
axs[0].set_title('Wasserstein Distance Affinity Matrix Histogram - BL')
axs[0].set_xlabel('Wasserstein Distance')
axs[0].set_ylabel('Count')

axs[1].stairs(hist_SD, edges_SD)
axs[1].set_title('Wasserstein Distance Affinity Matrix Histogram - SD')
axs[1].set_xlabel('Wasserstein Distance')
axs[1].set_ylabel('Count')

plt.show()

# import pdb; pdb.set_trace()

cmap = plt.get_cmap('tab10')

n_clusters_arr = np.array([2, 3, 4, 5])
# import pdb; pdb.set_trace()
BL_cluster_list = []
for n_clusters in n_clusters_arr:

    clustering = KMedoids(n_clusters=n_clusters,
                          metric='precomputed',
                          method = 'pam',
                          init = 'heuristic',    
                          random_state=None).fit(BL_dist_mat)
    group_assign = clustering.labels_
    group_assign = sort_groups_by_ah(BL_points, group_assign)
    np.save(f'../data/{dist_func_str}_BL_clusters_{n_clusters}.npy', group_assign)
    BL_cluster_list.append(group_assign)

SD_cluster_list = []
for n_clusters in n_clusters_arr:
    clustering = KMedoids(n_clusters=n_clusters,
                          metric='precomputed',
                          method = 'pam',
                          init = 'heuristic',
                          random_state=None).fit(SD_dist_mat)
    group_assign = clustering.labels_
    group_assign = sort_groups_by_ah(BL_points, group_assign)
    np.save(f'../data/{dist_func_str}_SD_clusters_{n_clusters}.npy', group_assign)
    SD_cluster_list.append(group_assign)

BL_cluster_list = [sort_groups_by_ah(BL_points, labels) for labels in BL_cluster_list]
SD_cluster_list = [sort_groups_by_ah(SD_points, labels) for labels in SD_cluster_list]

BL_silhouettes = [silhouette_samples(BL_dist_mat, labels, metric='precomputed') for labels in BL_cluster_list]
SD_silhouettes = [silhouette_samples(SD_dist_mat, labels, metric='precomputed') for labels in SD_cluster_list]

# import pdb; pdb.set_trace()

# for silhouettes_1 in BL_silhouettes:
#     for silhouettes_2 in SD_silhouettes:
#         fig = plt.figure(figsize=(14, 6))
#         BL_scatter_ax = fig.add_subplot(1, 2, 1, projection='3d')
#         SD_scatter_ax = fig.add_subplot(1, 2, 2, projection='3d')
#         plot_silhouettes(BL_points, silhouettes_1, [0, 1, 2], ['s', r'$a_h$', r'$a_l$', 'Area under SP'], 'BL Silhouette Scores', BL_scatter_ax)
#         plot_silhouettes(SD_points, silhouettes_2, [1, 2, 3], ['s', r'$a_h$', r'$a_l$', 'Area under SP'], 'SD Silhouette Scores', SD_scatter_ax)
#         plt.show()

# import pdb; pdb.set_trace()

for i, clusters_1 in enumerate(BL_cluster_list):
    for j, clusters_2 in enumerate(SD_cluster_list):
        fig = plt.figure(figsize=(14, 6))
        TM_ax = fig.add_subplot(1, 3, 1)
        BL_scatter_ax = fig.add_subplot(1, 3, 2, projection='3d')
        SD_scatter_ax = fig.add_subplot(1, 3, 3, projection='3d')
        plot_transition_matrix(clusters_1, clusters_2, TM_ax)
        plot_clusters(BL_points, clusters_1, [0, 1, 2], ['s', r'$a_h$', r'$a_l$', 'Area under SP'], BL_silhouettes[i], f'BL Clusters, sil = {np.mean(BL_silhouettes[i]):.2f}', BL_scatter_ax)
        plot_clusters(SD_points, clusters_2, [1, 2, 3], ['s', r'$a_h$', r'$a_l$', 'Area under SP'], SD_silhouettes[j], f'SD Clusters, sil = {np.mean(SD_silhouettes[j]):.2f}', SD_scatter_ax)
        plt.show()

