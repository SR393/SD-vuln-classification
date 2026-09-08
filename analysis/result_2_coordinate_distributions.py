import sys
sys.path.extend(['../data', '../analysis'])
import os
import re
from tqdm import tqdm

import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from scipy.stats import normaltest
from diptest import diptest
from dcor import independence
from sklearn.feature_selection import mutual_info_regression

import data_processing_funcs as dp
from log_pca_analysis_funcs import pdf_from_quantiles_kde
from block_length_estimation import ksg_mi_with_jitter

max_K = 8
M = 200
eps = 1e-3
alphas = np.linspace(eps, 1 - eps, M)

beta_BL = np.load(f'../data/logpca_coords_BL_{max_K}-dims.npy')
beta_BL[1] = -beta_BL[1] # reflect component 2 so negative values indicate poorer performance

beta_SD = np.load(f'../data/logpca_coords_SD_{max_K}-dims.npy')

unim_stat_BL_1, unim_pval_BL_1 = diptest(beta_BL[:, 0])
unim_stat_BL_2, unim_pval_BL_2 = diptest(beta_BL[:, 1])

unim_stat_SD_1, unim_pval_SD_1 = diptest(beta_SD[:, 0])
unim_stat_SD_2, unim_pval_SD_2 = diptest(beta_SD[:, 1])

print(unim_pval_BL_1, unim_pval_BL_2)
print(unim_pval_SD_1, unim_pval_SD_2)

ind_pval_BL, ind_stat_BL = independence.distance_covariance_test(beta_BL[:, 0], beta_BL[:, 1], num_resamples=1000)
ind_pval_SD, ind_stat_SD = independence.distance_covariance_test(beta_SD[:, 0], beta_SD[:, 1], num_resamples=1000)

# print(ind_pval_BL, ind_pval_SD)

mut_inf_BL = ksg_mi_with_jitter(beta_BL[:, 0], beta_BL[:, 1], k = 3, jitter = 1e-10, seed = 12345654321)
mut_inf_SD = ksg_mi_with_jitter(beta_SD[:, 0], beta_SD[:, 1], k = 3, jitter = 1e-10, seed = 12345654321)

print(mut_inf_BL, mut_inf_SD)

bootstrap_files = os.listdir('../data/bootstrap_samples/coordinates')

bootstrap_MI_BL = np.zeros(len(bootstrap_files)//2)
bootstrap_MI_SD = np.zeros(len(bootstrap_files)//2)

for file in tqdm(bootstrap_files, total = len(bootstrap_files)):
    index = int(re.split(r'[_ .]', file)[-2])
    betas = np.load(f'../data/bootstrap_samples/coordinates/{file}')
    betas_1 = betas[:, 0]/np.std(betas[:, 0])
    betas_2 = betas[:, 1]/np.std(betas[:, 1])
    mut_inf = ksg_mi_with_jitter(betas_1, betas_2, k = 3, jitter = 1e-10, seed = 12345654321)
    if 'BL' in file:
        bootstrap_MI_BL[index] = mut_inf
    else:
        bootstrap_MI_SD[index] = mut_inf

bootstrap_MI_BL_std = np.std(bootstrap_MI_BL)
bootstrap_MI_SD_std = np.std(bootstrap_MI_SD)

bs_mi_BL_z = np.zeros(len(bootstrap_MI_BL))
bs_mi_SD_z = np.zeros(len(bootstrap_MI_SD))
bs_mi_BL_z[bootstrap_MI_BL > 0] = bootstrap_MI_BL[bootstrap_MI_BL > 0]
bs_mi_SD_z[bootstrap_MI_SD > 0] = bootstrap_MI_SD[bootstrap_MI_SD > 0]

print(f'Bootstrap MI uncertainty (BL): {np.std(bs_mi_BL_z):.3e}')
print(f'Bootstrap MI uncertainty (SD): {np.std(bs_mi_SD_z):.3e}')

import pdb; pdb.set_trace()

component_1_BL_standardised_mag = np.abs(beta_BL[:, 0]/np.std(beta_BL[:, 0]))
component_2_BL_standardised_mag = np.abs(beta_BL[:, 1]/np.std(beta_BL[:, 1]))

component_1_SD_standardised_mag = np.abs(beta_SD[:, 0]/np.std(beta_SD[:, 0]))
component_2_SD_standardised_mag = np.abs(beta_SD[:, 1]/np.std(beta_SD[:, 1]))

component_1_BL_nonoutliers = np.nonzero(component_1_BL_standardised_mag < 2)[0]
component_2_BL_nonoutliers = np.nonzero(component_2_BL_standardised_mag < 2)[0]

component_1_SD_nonoutliers = np.nonzero(component_1_SD_standardised_mag < 2)[0]
component_2_SD_nonoutliers = np.nonzero(component_2_SD_standardised_mag < 2)[0]

stat_c1_BL, pval_c1_BL = normaltest(component_1_BL_standardised_mag[component_1_BL_nonoutliers])
stat_c2_BL, pval_c2_BL = normaltest(component_2_BL_standardised_mag[component_2_BL_nonoutliers])

stat_c1_SD, pval_c1_SD = normaltest(component_1_SD_standardised_mag[component_1_SD_nonoutliers])
stat_c2_SD, pval_c2_SD = normaltest(component_2_SD_standardised_mag[component_2_SD_nonoutliers])

import pdb; pdb.set_trace()