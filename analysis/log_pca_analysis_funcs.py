import sys
sys.path.extend(['../data'])
import os

import numpy as np
from KDEpy import FFTKDE
from sklearn.decomposition import PCA
from numba import njit, prange

from scipy.interpolate import PchipInterpolator

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

def reconstruct_quantiles(Qbar, components, coords):
    """
    Qbar: (M,) array of frechet mean quantiles
    components: (K, M) array of log-PCA components
    coords: (N, K) array of log-PCA coordinates
    returns: (N, M) array of reconstructed quantiles
    """
    Qbar = np.asarray(Qbar)
    components = np.asarray(components)
    coords = np.asarray(coords)

    Q_recon = Qbar + np.matmul(coords, components)
    N, K = coords.shape
    M = Qbar.shape[0]

    Q_recon = np.empty((N, M))
    for i in range(N):
        Q_recon[i] = Qbar + np.sum(coords[i, k] * components[k] for k in range(K))
    
    return Q_recon

@njit(parallel=True)
def draw_samples_from_quantiles(Q, alphas, n_samples=20000, eps=1e-3, rng = None):
    """
    Q: (M,) array of quantiles
    alphas: (M,) array of quantile levels in (0,1)
    returns: (n_samples,) array of samples drawn from the distribution defined by Q
    """
    # Q = np.asarray(Q)
    # alphas = np.asarray(alphas)

    # sample U in (eps, 1-eps) to avoid tail craziness
    if rng is None:
        u = np.random.uniform(eps, 1 - eps, size=n_samples)
    else:
        u = rng.uniform(eps, 1 - eps, size=n_samples)
    # import pdb; pdb.set_trace()
    samples = np.interp(u, alphas, Q)

    return samples

@njit(parallel=True)
def draw_samples_from_quantiles_arr(Q_arr, alphas, n_samples=20000, eps=1e-3, rngs = None):
    """
    Q_arr: (N, M) array of quantiles
    alphas: (M,) array of quantile levels in (0,1)
    returns: (N, n_samples) array of samples drawn from the distributions defined by each row of Q_arr
    """
    # Q_arr = np.asarray(Q_arr)
    # alphas = np.asarray(alphas)
    # import pdb; pdb.set_trace()
    N = Q_arr.shape[0]
    samples_arr = np.empty((N, n_samples))
    # if filedir is None or not os.path.exists(filedir):
        # samples_arr = np.array([draw_samples_from_quantiles(Q_arr[i], alphas, n_samples=n_samples, eps=eps) for i in range(N)])
    for i in prange(N):
        samples_arr[i] = draw_samples_from_quantiles(Q_arr[i], alphas, n_samples=n_samples, eps=eps, rng = rngs[i])
        # if filedir is not None:
            # np.save(filedir, samples_arr)
    # else:
        # samples_arr = np.load(filedir)
    
    return samples_arr

def pdf_from_quantiles_kde(Q, alphas, x_grid=None, samples = 20000, eps=1e-3):

    """
    Q: (M,) array of quantiles
    alphas: (M,) array of quantile levels in (0,1)
    returns: (L,) array of densities on x_grid
    """

    # if isinstance(samples, int):
    #     samples = draw_samples_from_quantiles(Q, alphas, n_samples=samples, eps=eps)

    if x_grid is None:
        lo, hi = np.quantile(samples, [0.001, 0.999])
        x_grid = np.linspace(lo, hi, 400)

    # kde = gaussian_kde(samples)
    # pdf = fastkde.pdf_at_points(samples, list_of_points = x_grid)
    al_int = np.linspace(alphas[0], alphas[-1], 10000)
    Q_int = PchipInterpolator(alphas, Q).__call__(al_int)
    pdf = FFTKDE(bw='silverman').fit(Q_int).evaluate(x_grid)
    return x_grid, pdf # kde(x_grid) 

def pdf_from_quantiles_arr(Q_arr, alphas, x_grid=None, samples=20000, eps=1e-3, filedir = None):
    """
    Q_arr: (N, M) array of quantiles
    returns: (N, L) array of densities on x_grid
    """

    if filedir is None or not os.path.exists(filedir):
        if isinstance(samples, int):
            densities = np.array([pdf_from_quantiles_kde(Q, alphas, x_grid=x_grid, samples=samples, eps=eps)[1] for Q in Q_arr])
        else:
            densities = np.array([pdf_from_quantiles_kde(Q, alphas, x_grid=x_grid, samples=samples[i], eps=eps)[1] for i, Q in enumerate(Q_arr)])
        if filedir is not None:
            np.save(filedir, densities)
    else:
        densities = np.load(filedir)

    return x_grid, densities

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

# def estimate_empirical_peaks(pdf, data, x):

#     zeros, zero_inds = dp.get_derivative_zeros(x, pdf, return_inds = True)
#     pdf_vals = pdf[zero_inds]

#     if len(zeros) == 1:
#         return zeros, np.array([1.0])
#     elif len(zeros) % 2 == 0:
#         raise ValueError('Number of pdf turning points should be odd.')
#     elif pdf_vals[0] > pdf_vals[1]:
#         peak_locs = zeros[::2]
#         bounds = np.concatenate((np.array([np.min(x)]), zeros[1::2], np.array([np.max(x)])))
#     else:
#         peak_locs = zeros[1::2]
#         bounds = np.concatenate((np.array([np.min(x)]), zeros[::2], np.array([np.max(x)])))
    
#     bounds = np.array([bounds[:-1], bounds[1:]]).T
#     proportions = np.array([np.sum((data > lb)*(data <= ub)) for lb, ub in bounds])/len(data)

#     return peak_locs, proportions

# @njit(parallel=True)
def estimate_empirical_peaks(pdf, x, data = None):

    bounds = []
    peaks = []

    # only include edges as bounds if they are not local maxima; otherwise, exclude from consideration
    if pdf[0] <= pdf[1]:
        bounds.append(np.array([x[0]]))
    if pdf[-1] <= pdf[-2]:
        bounds.append(np.array([x[-1]]))
    
    forward_diff = np.diff(pdf[:-1])
    backward_diff = -np.diff(pdf[1:])
    if np.nonzero((forward_diff < 0)*(backward_diff < 0))[0].size > 0:
        bounds.append(x[np.nonzero((forward_diff < 0)*(backward_diff < 0))[0] + 1]) # inner bounds are local minima between peaks
    peaks.append(x[np.nonzero((forward_diff > 0)*(backward_diff > 0))[0] + 1])
      
    # bounds can be missed because of plateau regions in pdf; if so, find plateau regions and set bounds at their midpoints
    if not len(np.concatenate(bounds)) == len(np.concatenate(peaks)) + 1:
        peaks_ = np.sort(np.concatenate(peaks))
        neighbouring_peak_pairs = np.array([peaks_[:-1], peaks_[1:]]).T

        for pair in neighbouring_peak_pairs:

            plateau_inds = np.nonzero((x[1:-1] > pair[0])*(x[1:-1] < pair[1])*(forward_diff <= 0)*(backward_diff <= 0))[0] + 1
            midpoint = plateau_inds[len(plateau_inds) // 2]
            if x[midpoint] in np.concatenate(bounds):
                continue
            bounds.append(np.array([x[midpoint]]))

    bounds = np.sort(np.concatenate(bounds))
    peaks = np.sort(np.concatenate(peaks))
    bounds = np.array([bounds[:-1], bounds[1:]]).T 
    if not data is None:   
        proportions = np.array([np.sum((data > lb)*(data <= ub)) for lb, ub in bounds])/len(data)
    else:
        # import pdb; pdb.set_trace()
        bound_inds = np.array([np.searchsorted(x, bounds[:, 0]), np.searchsorted(x, bounds[:, 1])])
        proportions = np.array([np.trapz(pdf[b1:b2], x[b1:b2]) for b1, b2 in bound_inds.T])

    return peaks, proportions

# @njit(parallel=True)
def extract_prim_and_sec_peak(peak_locs, proportions, thresh = None):

    # initial guesses; primary peak contains the most observations, secondary peak is the lowest peak
    primary_peak = peak_locs[np.argmax(proportions)]
    secondary_peak = np.min(peak_locs)

    if len(peak_locs) == 1: # if there's only one peak, determine whether it is primary or secondary based on threshold
        if thresh is None or peak_locs[0] > thresh:
            peaks = np.array([np.nan, peak_locs[0]])
            props = np.array([0.0, 1.0])
        else:
            peaks = np.array([peak_locs[0], np.nan])
            props = np.array([1.0, 0.0])

    elif primary_peak == secondary_peak: # if there is more than one peak, but the lowest peak contains the most observations
        # In this case, the lower peak may be a primary peak, as small peaks can appear at very high values (~ 5.0 s^-1)
        # in which case a lower peak around 1.5 - 4.0 s^-1 should usually be considered the primary peak
        
        if thresh is None or primary_peak >= thresh: # if the peak is higher than the threshold or no threshold is used, call it the primary peak
            peaks = np.array([np.nan, primary_peak])
            props = np.array([np.nan, np.max(proportions)])
        else: # if the peak is "low enough" (< thresh ~= 1.0; group-aggregate secondary peaks tend to be around 0.25 s^-1), call it the secondary peak
            proportions_order = np.argsort(proportions)
            try:
                second_largest_peak_ind = proportions_order[-2]
            except IndexError:
                import pdb; pdb.set_trace()
                second_largest_peak_ind = proportions_order[-1]
            second_largest_peak_ind = proportions_order[-2]
            peaks = np.array([secondary_peak, peak_locs[second_largest_peak_ind]]) # define the primary peak as the peak with the second-most observations
            props = np.array([np.max(proportions), np.sort(proportions)[-2]]) # and the secondary peak as the peak with the most observations
            
    else: # otherwise, assume initial guesses are good
        peaks = np.array([secondary_peak, primary_peak])
        props = np.array([proportions[np.argmin(peak_locs)], np.max(proportions)])
    
    return peaks, props

# @njit(parallel=True)
def get_empirical_estimates(pdf, x, data = None, thresh = 1.0):

    peak_locs, proportions = estimate_empirical_peaks(pdf, x, data)
    peak_locs, proportions = extract_prim_and_sec_peak(peak_locs, proportions, thresh = thresh)

    return peak_locs, proportions

# @njit(parallel=True)
def get_empirical_peaks_all(x, pdfs, samples = None):

    primary_peak_locs = np.empty(len(pdfs))*np.nan
    secondary_peak_locs = np.empty(len(pdfs))*np.nan
    primary_peak_proportions = np.empty(len(pdfs))*np.nan
    secondary_peak_proportions = np.empty(len(pdfs))*np.nan
    if not samples is None:
        for i, (pdf, rRTs) in enumerate(zip(pdfs, samples)):

            peak_locs, proportions = get_empirical_estimates(pdf, x, rRTs, thresh = 1.0)

            primary_peak_locs[i] = peak_locs[1]
            secondary_peak_locs[i] = peak_locs[0]
            primary_peak_proportions[i] = proportions[1]
            secondary_peak_proportions[i] = proportions[0]
    else:
        for i, pdf in enumerate(pdfs):

            peak_locs, proportions = get_empirical_estimates(pdf, x, thresh = 1.0)

            primary_peak_locs[i] = peak_locs[1]
            secondary_peak_locs[i] = peak_locs[0]
            primary_peak_proportions[i] = proportions[1]
            secondary_peak_proportions[i] = proportions[0]

    return primary_peak_locs, secondary_peak_locs, primary_peak_proportions, secondary_peak_proportions