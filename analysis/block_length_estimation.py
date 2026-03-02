import sys
sys.path.extend(['../data'])
import os
from tqdm import tqdm

import numpy as np
from numba import njit, prange

import data_processing_funcs as dp

@njit(cache=True)
def _digamma(x: float) -> float:
    """
    Approximation of digamma psi(x) for x > 0.

    Uses recurrence to shift to x >= 8 and then asymptotic expansion.
    Accuracy is more than adequate for KSG with small k and n ~ 50-2000.
    """
    # Handle small x
    if x <= 0.0:
        return np.nan

    result = 0.0
    # Shift via recurrence to use asymptotic expansion for large x
    while x < 8.0:
        result -= 1.0 / x
        x += 1.0

    inv = 1.0 / x
    inv2 = inv * inv

    # Asymptotic series: psi(x) = log(x) - 1/(2x) - 1/(12x^2) + 1/(120x^4) - 1/(252x^6) + ...
    result += np.log(x) - 0.5 * inv - (1.0 / 12.0) * inv2
    inv4 = inv2 * inv2
    result += (1.0 / 120.0) * inv4
    inv6 = inv4 * inv2
    result -= (1.0 / 252.0) * inv6

    return result

@njit(parallel=True, cache=True)
def ksg_mi_1d_1d(x: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """
    k-nearest neighbors mutual information estimator (ref)
    for two 1D continuous variables.
    """

    n = x.shape[0]

    # Arrays of neighbor counts for each i
    nx = np.empty(n, dtype=np.int64)
    ny = np.empty(n, dtype=np.int64)

    for i in prange(n):
        xi = x[i]
        yi = y[i]

        # Find k-th nearest neighbor distance in joint (max-norm), excluding self.
        # Since n is small (~90), we do O(n) scanning with a small "top-k" array.
        # Keep the k smallest distances seen so far in a fixed array.
        topk = np.full(k, np.inf, dtype=np.float64)

        for j in range(n):
            if j == i:
                continue
            dx = abs(x[j] - xi)
            dy = abs(y[j] - yi)
            d = dx if dx > dy else dy  # max-norm

            # Insert d into topk if smaller than current max in topk.
            # This is O(k) per insert; with small k it's fine.
            # Find current largest in topk:
            max_idx = 0
            max_val = topk[0]
            for t in range(1, k):
                if topk[t] > max_val:
                    max_val = topk[t]
                    max_idx = t

            if d < max_val:
                topk[max_idx] = d

        # eps is the maximum of the k smallest distances (k-th neighbor distance)
        eps = topk[0]
        for t in range(1, k):
            if topk[t] > eps:
                eps = topk[t]

        if eps < 0.0:
            eps = 0.0

        # Count neighbors within eps
        cx = 0
        cy = 0
        for j in range(n):
            if j == i:
                continue
            if abs(x[j] - xi) <= eps:
                cx += 1
            if abs(y[j] - yi) <= eps:
                cy += 1

        nx[i] = cx
        ny[i] = cy

    # MI = psi(k) + psi(n) - mean(psi(nx+1) + psi(ny+1))

    psi_k = _digamma(float(k))
    psi_n = _digamma(float(n))

    s = 0.0
    for i in range(n):
        s += _digamma(float(nx[i] + 1)) + _digamma(float(ny[i] + 1))
    s /= n

    mi = psi_k + psi_n - s
    return mi


def ksg_mi_with_jitter(x: np.ndarray, y: np.ndarray, k: int = 5, jitter: float = 0.0, seed: int = 0) -> float:
    """
    Wrapper that adds small Gaussian perturbations to break ties.
    Jitter ~ 1e-10 * std(x).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if jitter > 0.0:
        rng = np.random.default_rng(seed)
        sx = np.std(x)
        sy = np.std(y)
        x = x + rng.normal(0.0, jitter * (sx if sx > 0 else 1.0), size=x.shape[0])
        y = y + rng.normal(0.0, jitter * (sy if sy > 0 else 1.0), size=y.shape[0])
    return float(ksg_mi_1d_1d(x, y, k))

def individual_MI_values(sessions, n_permutations, n_lags, seed):

    true_MIs = np.zeros((n_lags, len(sessions)))*np.nan
    null_MIs = np.zeros((n_lags, len(sessions), n_permutations))*np.nan
    rng = np.random.default_rng(seed)
    for s_ind, session in enumerate(sessions):
        if len(session) == 0:
            continue
        for i in range(1, n_lags + 1):
            prec_rRTs = 1000 / session[:-i]
            succ_rRTs = 1000 / session[i:]
            MI = ksg_mi_with_jitter(prec_rRTs, succ_rRTs, k = 4, jitter = 1e-10, seed = seed)
            true_MIs[i-1, s_ind] = MI
            for p in range(n_permutations):
                succ_rRTs_perm = rng.permutation(succ_rRTs)
                MI_perm = ksg_mi_with_jitter(prec_rRTs, succ_rRTs_perm, k = 4, jitter = 1e-10, seed = seed)
                null_MIs[i-1, s_ind, p] = MI_perm
    return true_MIs, null_MIs

def estimate_p_value(true_MI, null_MIs):
    p_values = np.mean(null_MIs <= true_MI)
    return p_values

if __name__ == "__main__":

    RTs_dict = dp.load_RT_data_dict()
    RTs_dict = dp.filter_data(RTs_dict, lower_cutoff = 167, upper_cutoff = 10000, genotype = None, remove_adaptation_effect = True, remove_circ_misaligned = True)

    rng = np.random.default_rng(seed=12345)
    seeds = rng.integers(2**32 - 1, size = len(RTs_dict))  # Generate a seed for the block length estimation process

    n_lags = 10
    n_permutations = 1000
    
    if not os.path.exists('../data/block_length_estimation_true_MIs.npy'):
        true_MIs = np.zeros((n_lags, len(RTs_dict), 20))*np.nan
        null_MIs = np.zeros((n_lags, len(RTs_dict), 20, n_permutations))*np.nan
        p_values = np.zeros((n_lags, len(RTs_dict), 20))*np.nan
        for i, sessions in tqdm(enumerate(RTs_dict.values()), total=len(RTs_dict)):
            seed = rng.integers(2**32 - 1)
            true_MI_ind, null_MIs_ind = individual_MI_values(sessions, n_permutations, n_lags, seed)
            p_values_ind = np.array([[estimate_p_value(true_MI, null_MIs) for true_MI, null_MIs in zip(true_MI_ind[j], null_MIs_ind[j])] for j in range(n_lags)])
            p_values[:, i, :] = p_values_ind
            for j in range(n_lags):
                true_MIs[j, i, :] = true_MI_ind[j]
                null_MIs[j, i, :, :] = null_MIs_ind[j]
            # if i == 5:
            #     import pdb; pdb.set_trace()
            # print(f'individual {i} complete')
        np.save('../data/block_length_estimation_true_MIs.npy', true_MIs)
        np.save('../data/block_length_estimation_null_MIs.npy', null_MIs)
        np.save('../data/block_length_estimation_p_values.npy', p_values)
    else:
        true_MIs = np.load('../data/block_length_estimation_true_MIs.npy')
        null_MIs = np.load('../data/block_length_estimation_null_MIs.npy')
        p_values = np.load('../data/block_length_estimation_p_values.npy')

    print(np.mean(np.mean(p_values, axis = 1), axis = 1))