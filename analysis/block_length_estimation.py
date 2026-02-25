import sys
sys.path.extend(['../data'])
import multiprocessing as mproc
from functools import partial

import numpy as np
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt

import data_processing_funcs as dp

import numpy as np
from numba import njit, prange


# ---------------------------
# Digamma approximation (Numba-friendly)
# ---------------------------
@njit(cache=True)
def _digamma(x: float) -> float:
    """
    Approximation of digamma psi(x) for x > 0.

    Uses recurrence to shift to x >= 8 and then asymptotic expansion.
    Accuracy is more than adequate for KSG with small k and n ~ 50-2000.
    """
    # Handle tiny x (shouldn't happen in our use: we call digamma(n+1) etc.)
    if x <= 0.0:
        return np.nan

    result = 0.0
    # Shift upward
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


# ---------------------------
# Core: KSG MI estimator for 1D-1D variables
# ---------------------------
@njit(parallel=True, cache=True)
def ksg_mi_1d_1d(x: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """
    Kraskov–Stögbauer–Grassberger mutual information estimator (KSG-1)
    for two 1D continuous variables.

    Parameters
    ----------
    x, y : 1D arrays of same length n
    k    : number of neighbors (typical 3..10)

    Returns
    -------
    mi : float
        Estimated MI in nats (natural log).
        Can be slightly negative in finite samples; treat tiny negatives as ~0.
    """
    n = x.shape[0]
    if y.shape[0] != n:
        raise ValueError("x and y must have same length")
    if k < 1 or k >= n:
        raise ValueError("k must satisfy 1 <= k < n")

    # Arrays of neighbor counts for each i
    nx = np.empty(n, dtype=np.int64)
    ny = np.empty(n, dtype=np.int64)

    # A tiny offset to avoid counting points at exactly eps due to ties
    tiny = 1e-15

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
        eps -= tiny
        if eps < 0.0:
            eps = 0.0

        # Count neighbors in marginal spaces within eps
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
    # Some references include a -1/k term for certain variants; the common KSG-1 form is below.
    psi_k = _digamma(float(k))
    psi_n = _digamma(float(n))

    s = 0.0
    for i in range(n):
        s += _digamma(float(nx[i] + 1)) + _digamma(float(ny[i] + 1))
    s /= n

    mi = psi_k + psi_n - s
    return mi


# ---------------------------
# Convenience: jitter to break ties (recommended for quantized RTs)
# ---------------------------
def ksg_mi_with_jitter(x: np.ndarray, y: np.ndarray, k: int = 5, jitter: float = 0.0, seed: int = 0) -> float:
    """
    Wrapper that optionally adds tiny Gaussian jitter (in Python) to break ties.
    Use jitter ~ 1e-10 * std(x) typically.
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
            # MI = mutual_info_regression(prec_rRTS.reshape(-1, 1), succ_rRTS, random_state = seed)[0]
            MI = ksg_mi_with_jitter(prec_rRTS, succ_rRTS, k = 5, jitter = 1e-10, seed = seed)
            true_MIs[i-1, s_ind] = MI
            for p in range(n_permutations):
                session_perm = permutations[p]
                prec_rRTs_perm = 1000 / session_perm[:-i]
                succ_rRTs_perm = 1000 / session_perm[i:]
                # MI_perm = mutual_info_regression(prec_rRTs_perm.reshape(-1, 1), succ_rRTs_perm, random_state = seed)[0]
                MI_perm = ksg_mi_with_jitter(prec_rRTs_perm, succ_rRTs_perm, k = 5, jitter = 1e-10, seed = seed)
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

    n_lags = 10
    n_permutations = 1000
    
    # ind_MI_partial = partial(individual_MI_values, n_permutations = n_permutations, n_lags = n_lags)

    # with mproc.Pool(ncpus) as pool:
    #     results = pool.starmap(ind_MI_partial, zip(RTs_dict.values(), seeds))

    true_MIs = []
    null_MIs = []

    for i, sessions in enumerate(RTs_dict.values()):
        seed = rng.integers(2**32 - 1)
        true_MI_ind, null_MIs_ind = individual_MI_values(sessions, n_permutations, n_lags, seed)
        true_MIs.append(true_MI_ind)
        null_MIs.append(null_MIs_ind)

    import pdb; pdb.set_trace()

    for i in range(n_lags):
        true_MIs = np.array([r[0][i] for r in results])
        null_MIs = np.array([r[1][i] for r in results])
        p_values = np.array([estimate_p_value(true_MI, null_MIs) for true_MI, null_MIs in zip(true_MIs, null_MIs)])

    import pdb; pdb.set_trace()