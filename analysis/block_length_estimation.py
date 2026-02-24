import sys
sys.path.extend(['../data'])
import multiprocessing as mproc
from functools import partial

import numpy as np
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt

import data_processing_funcs as dp

def individual_MI_values(sessions, n_permutations, n_lags, seed):

    true_MIs = np.zeros((n_lags, len(sessions)))
    null_MIs = np.zeros((n_lags, len(sessions), n_permutations))
    rng = np.random.default_rng(seed)
    for s_ind, session in enumerate(sessions):
        permutations = [rng.permutation(session) for _ in range(n_permutations)]
        if len(session) == 0:
            continue
        session_perm = rng.permutation(session)
        for i in range(1, n_lags + 1):
            prec_rRTS = 1000 / session[:-i]
            succ_rRTS = 1000 / session[i:]
            MI = mutual_info_regression(prec_rRTS.reshape(-1, 1), succ_rRTS, random_state = seed)[0]
            true_MIs[i-1, s_ind] = MI
            for p in range(n_permutations):
                session_perm = permutations[p]
                prec_rRTs_perm = 1000 / session_perm[:-i]
                succ_rRTs_perm = 1000 / session_perm[i:]
                MI_perm = mutual_info_regression(prec_rRTs_perm.reshape(-1, 1), succ_rRTs_perm, random_state = seed)[0]
                null_MIs[i-1, s_ind, p] = MI_perm

    return true_MIs, null_MIs

def estimate_p_value(true_MI, null_MIs):
    p_values = np.mean(null_MIs <= true_MI)
    return p_values

if __name__ == "__main__":

    ncpus = 20

    RTs_dict = dp.load_RT_data_dict()
    RTs_dict = dp.filter_data(RTs_dict, lower_cutoff = 167, upper_cutoff = 10000, genotype = None, remove_adaptation_effect = True, remove_circ_misaligned = True)

    rng = np.random.default_rng(seed=12345)
    seeds = rng.integers(2**32 - 1, size = len(RTs_dict))  # Generate a seed for the block length estimation process
    ### Mutual Information Approach ###

    n_lags = 20
    n_permutations = 1000
    
    ind_MI_partial = partial(individual_MI_values, n_permutations = n_permutations, n_lags = n_lags)

    with mproc.Pool(ncpus) as pool:
        results = pool.starmap(ind_MI_partial, zip(RTs_dict.values(), seeds))
    
    for i in range(n_lags):
        true_MIs = np.array([r[0][i] for r in results])
        null_MIs = np.array([r[1][i] for r in results])
        p_values = np.array([estimate_p_value(true_MI, null_MIs) for true_MI, null_MIs in zip(true_MIs, null_MIs)])

    import pdb; pdb.set_trace()