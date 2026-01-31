import sys
sys.path.extend(['../data'])
import os

import numpy as np
from scipy.stats import gaussian_kde
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

def draw_samples_from_quantiles(Q, alphas, n_samples=20000, eps=1e-3):
    """
    Q: (M,) array of quantiles
    alphas: (M,) array of quantile levels in (0,1)
    returns: (n_samples,) array of samples drawn from the distribution defined by Q
    """
    Q = np.asarray(Q)
    alphas = np.asarray(alphas)

    # sample U in (eps, 1-eps) to avoid tail craziness
    u = np.random.uniform(eps, 1 - eps, size=n_samples)
    # import pdb; pdb.set_trace()
    samples = np.interp(u, alphas, Q)
    return samples

def pdf_from_quantiles_kde(Q, alphas, x_grid=None, n_samples=20000, eps=1e-3):

    """
    Q: (M,) array of quantiles
    alphas: (M,) array of quantile levels in (0,1)
    returns: (L,) array of densities on x_grid
    """
    
    samples = draw_samples_from_quantiles(Q, alphas, n_samples=n_samples, eps=eps)

    if x_grid is None:
        lo, hi = np.quantile(samples, [0.001, 0.999])
        x_grid = np.linspace(lo, hi, 400)

    kde = gaussian_kde(samples)
    return x_grid, kde(x_grid)

def pdf_from_quantiles_arr(Q_arr, alphas, x_grid=None, n_samples=20000, eps=1e-3, filedir = None):
    """
    Q_arr: (N, M) array of quantiles
    returns: (N, L) array of densities on x_grid
    """

    if filedir is None or not os.path.exists(filedir):
        densities = np.array([pdf_from_quantiles_kde(Q, alphas, x_grid=x_grid, n_samples=n_samples, eps=eps)[1] for Q in Q_arr])
        if filedir is not None:
            np.save(filedir, densities)
        return x_grid, densities
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

def estimate_empirical_peaks(pdf, data, x):

    zeros, zero_inds = dp.get_derivative_zeros(x, pdf, return_inds = True)
    pdf_vals = pdf[zero_inds]

    if len(zeros) == 1:
        return zeros, np.array([1.0])
    elif len(zeros) % 2 == 0:
        raise ValueError('Number of pdf turning points should be odd.')
    elif pdf_vals[0] > pdf_vals[1]:
        peak_locs = zeros[::2]
        bounds = np.concatenate((np.array([np.min(x)]), zeros[1::2], np.array([np.max(x)])))
    else:
        peak_locs = zeros[1::2]
        bounds = np.concatenate((np.array([np.min(x)]), zeros[::2], np.array([np.max(x)])))
    
    bounds = np.array([bounds[:-1], bounds[1:]]).T
    proportions = np.array([np.sum((data > lb)*(data <= ub)) for lb, ub in bounds])/len(data)

    return peak_locs, proportions

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
            second_largest_peak_ind = proportions_order[-2]
            peaks = np.array([secondary_peak, peak_locs[second_largest_peak_ind]]) # define the primary peak as the peak with the second-most observations
            props = np.array([np.max(proportions), np.sort(proportions)[-2]]) # and the secondary peak as the peak with the most observations
            
    else: # otherwise, assume initial guesses are good
        peaks = np.array([secondary_peak, primary_peak])
        props = np.array([proportions[np.argmin(peak_locs)], np.max(proportions)])
    
    return peaks, props

def get_empirical_estimates(pdf, data, x, thresh = 1.0):

    peak_locs, proportions = estimate_empirical_peaks(pdf, data, x)
    peak_locs, proportions = extract_prim_and_sec_peak(peak_locs, proportions, thresh = thresh)

    return peak_locs, proportions

def get_empirical_peaks_all(x, individuals, RTs_dict, times_dict, trough_loc, T_bounds):

    primary_peak_locs = np.empty((len(individuals), T_bounds.shape[0]))*np.nan
    secondary_peak_locs = np.empty((len(individuals), T_bounds.shape[0]))*np.nan
    primary_peak_proportions = np.empty((len(individuals), T_bounds.shape[0]))*np.nan
    secondary_peak_proportions = np.empty((len(individuals), T_bounds.shape[0]))*np.nan

    for i, ind in enumerate(individuals):

        individual_times = (times_dict[ind]['time_arr'] - times_dict[ind]['DLMO'])[1:] # times relative to DLMO; remove wake-up time
        DLMO_session_mask = np.array([(bounds[0] < individual_times)*(bounds[1] >= individual_times) for bounds in T_bounds])

        for DLMO_t_ind, session_mask_t in enumerate(DLMO_session_mask):
            session_inds = np.nonzero(session_mask_t)[0]
            if len(session_inds) == 0:
                continue
            rRTs = np.concatenate([1/RTs_dict[ind][t]*1000 for t in session_inds])

            pdf = gaussian_kde(rRTs).evaluate(x)
            peak_locs, proportions = get_empirical_estimates(pdf, rRTs, x, thresh = 1.0)

            primary_peak_locs[i, DLMO_t_ind] = peak_locs[1]
            secondary_peak_locs[i, DLMO_t_ind] = peak_locs[0]
            primary_peak_proportions[i, DLMO_t_ind] = proportions[1]
            secondary_peak_proportions[i, DLMO_t_ind] = proportions[0]

    return primary_peak_locs, secondary_peak_locs, primary_peak_proportions, secondary_peak_proportions