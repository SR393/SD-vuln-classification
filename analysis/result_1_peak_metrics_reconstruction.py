import sys
sys.path.extend(['../data'])
import os
import json
from tqdm import tqdm

import numpy as np
from KDEpy import FFTKDE

import data_processing_funcs as dp
from log_pca_analysis_funcs import (
    geodesic_quantiles_from_component,
    pdf_from_quantiles_arr,
    get_empirical_peaks_all,
    reconstruct_quantiles,
)

def get_peak_width(x_grid, density, peak_loc, height_frac=0.5):
    peak_height = np.max(density)
    half_height = peak_height * height_frac

    left_mask = x_grid < peak_loc
    right_mask = x_grid > peak_loc

    left_x = x_grid[left_mask]
    left_d = density[left_mask]
    right_x = x_grid[right_mask]
    right_d = density[right_mask]

    left_cross = np.where(left_d <= half_height)[0]
    left_width = peak_loc - (left_x[left_cross[-1]] if len(left_cross) else x_grid[0])

    right_cross = np.where(right_d <= half_height)[0]
    right_width = (right_x[right_cross[0]] if len(right_cross) else x_grid[-1]) - peak_loc

    return left_width + right_width


def peak_widths_from_densities(x_grid, densities, peak_locs):
    return np.array([
        get_peak_width(x_grid, d, loc)
        for d, loc in zip(densities, peak_locs)
    ])


def corr(a, b):
    return np.corrcoef(a, b)[0, 1]

def compute_peak_metrics(rRTs_BL, rRTs_SD,
                         Qbar_BL, components_BL, coords_BL,
                         Qbar_SD, components_SD, coords_SD,
                         alphas, x_grid, x1_ind):
    """
    Returns dict of metrics comparing empirical (FFTKDE) peaks to
    reconstructed peaks for comp1, comp2, comp1+2 for BL and SD.
    Secondary peak analysis is SD-only.
    """
    # --- Empirical KDEs ---
    emp_kdes_BL = np.array([FFTKDE(bw='silverman').fit(r).evaluate(x_grid) for r in rRTs_BL])
    emp_kdes_SD = np.array([FFTKDE(bw='silverman').fit(r).evaluate(x_grid) for r in rRTs_SD])

    emp_pp_locs_BL, emp_sp_locs_BL, emp_pp_props_BL, _ = get_empirical_peaks_all(x_grid, emp_kdes_BL, rRTs_BL)
    emp_pp_locs_SD, emp_sp_locs_SD, emp_pp_props_SD, _ = get_empirical_peaks_all(x_grid, emp_kdes_SD, rRTs_SD)

    emp_pp_widths_BL = peak_widths_from_densities(x_grid, emp_kdes_BL, emp_pp_locs_BL)
    emp_pp_widths_SD = peak_widths_from_densities(x_grid, emp_kdes_SD, emp_pp_locs_SD)

    # Left-tail mass (used in secondary peak analysis as proxy for SP size)
    emp_left_tail_BL = np.array([np.trapz(k[:x1_ind], x_grid[:x1_ind]) for k in emp_kdes_BL])
    emp_left_tail_SD = np.array([np.trapz(k[:x1_ind], x_grid[:x1_ind]) for k in emp_kdes_SD])

    n_obs_BL = np.array([len(r) for r in rRTs_BL])
    n_obs_SD = np.array([len(r) for r in rRTs_SD])

    results = {}

    # Per-condition reconstruction loop
    for cond, Qbar, components, coords, emp_pp_locs, emp_pp_widths, emp_pp_props, emp_sp_locs, emp_left_tail, n_obs in [
        ('BL', Qbar_BL, components_BL, coords_BL,
         emp_pp_locs_BL, emp_pp_widths_BL, emp_pp_props_BL, emp_sp_locs_BL, emp_left_tail_BL, n_obs_BL),
        ('SD', Qbar_SD, components_SD, coords_SD,
         emp_pp_locs_SD, emp_pp_widths_SD, emp_pp_props_SD, emp_sp_locs_SD, emp_left_tail_SD, n_obs_SD),
    ]:
        # Build quantile arrays for each model
        Q_c1  = np.array(geodesic_quantiles_from_component(Qbar, components[0], t_vals=coords[:, 0])[1])
        Q_c2  = np.array(geodesic_quantiles_from_component(Qbar, components[1], t_vals=coords[:, 1])[1])
        Q_c12 = reconstruct_quantiles(Qbar, components[:2], coords[:, :2])

        for label, Q in [('c1', Q_c1), ('c2', Q_c2), ('c12', Q_c12)]:
            densities = pdf_from_quantiles_arr(Q, alphas, x_grid=x_grid)[1]
            pp_locs, sp_locs, _, _ = get_empirical_peaks_all(x_grid, densities)

            # Primary peak location
            results[f'{cond}_{label}_r_pp_loc'] = corr(emp_pp_locs, pp_locs)

            # Primary peak width
            pp_widths = peak_widths_from_densities(x_grid, densities, pp_locs)
            results[f'{cond}_{label}_r_pp_width']    = corr(emp_pp_widths, pp_widths)

            # Secondary peak (SD only)
            if cond == 'SD':
                left_tail = np.array([np.trapz(d[:x1_ind], x_grid[:x1_ind]) for d in densities])

                both_have_sp = (~np.isnan(emp_sp_locs)) & (~np.isnan(sp_locs))
                prop_both = np.sum(both_have_sp) / len(both_have_sp)
                idx = np.nonzero(both_have_sp)[0]

                if len(idx) >= 2:
                    r_sp_loc       = corr(sp_locs[idx], emp_sp_locs[idx])
                else:
                    r_sp_loc    = np.nan

                results[f'SD_{label}_sp_prop_both']       = prop_both
                results[f'SD_{label}_r_sp_loc']           = r_sp_loc

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    K = 8
    eps = 1e-3
    M = 200
    alphas = np.linspace(eps, 1 - eps, M)

    x_grid = np.linspace(-1.0, 6.5, 1000)
    x1_ind = int(np.searchsorted(x_grid, 1.0))

    # Original data
    rRTs_1, rRTs_2 = dp.get_standard_individual_rRT_data()

    Qbar_BL       = np.load(f'../data/logpca_frechet_mean_BL_{K}-dims.npy')
    components_BL = np.load(f'../data/logpca_components_BL_{K}-dims.npy')
    coords_BL     = np.load(f'../data/logpca_coords_BL_{K}-dims.npy')

    Qbar_SD       = np.load(f'../data/logpca_frechet_mean_SD_{K}-dims.npy')
    components_SD = np.load(f'../data/logpca_components_SD_{K}-dims.npy')
    coords_SD     = np.load(f'../data/logpca_coords_SD_{K}-dims.npy')
    
    # Apply BL component 2 sign flip (consistent with other analyses)
    coords_BL = coords_BL.copy()
    coords_BL[:, 1] = -coords_BL[:, 1]
    components_BL = components_BL.copy()
    components_BL[1] = -components_BL[1]

    true_metrics = compute_peak_metrics(
        rRTs_1, rRTs_2,
        Qbar_BL, components_BL, coords_BL,
        Qbar_SD, components_SD, coords_SD,
        alphas, x_grid, x1_ind
    )


    # Bootstrap
    bs_dir = '../data/bootstrap_samples'
    rt_files = sorted(os.listdir(os.path.join(bs_dir, 'RTs')))

    all_bs_metrics = []
    for fname in tqdm(rt_files, desc='bootstrap'):
        rts_dict = dp.json_to_dict(os.path.join(bs_dir, 'RTs', fname))
        idx = rts_dict['bs_index']
        del rts_dict['bs_index']
        rts_dict = {pid: [np.array(s) for s in sessions] for pid, sessions in rts_dict.items()}

        bs_rRTs_1, bs_rRTs_2 = dp.get_standard_individual_rRT_data(rts_dict)

        bs_Qbar_BL       = np.load(os.path.join(bs_dir, f'q_means/logpca_frechet_mean_BL_{K}-dims_{idx}.npy'))
        bs_components_BL = np.load(os.path.join(bs_dir, f'components/logpca_components_BL_{K}-dims_{idx}.npy'))
        bs_coords_BL     = np.load(os.path.join(bs_dir, f'coordinates/logpca_coords_BL_{K}-dims_{idx}.npy'))

        bs_Qbar_SD       = np.load(os.path.join(bs_dir, f'q_means/logpca_frechet_mean_SD_{K}-dims_{idx}.npy'))
        bs_components_SD = np.load(os.path.join(bs_dir, f'components/logpca_components_SD_{K}-dims_{idx}.npy'))
        bs_coords_SD     = np.load(os.path.join(bs_dir, f'coordinates/logpca_coords_SD_{K}-dims_{idx}.npy'))

        # Apply BL component 2 sign flip
        bs_coords_BL = bs_coords_BL.copy()
        bs_coords_BL[:, 1] = -bs_coords_BL[:, 1]
        bs_components_BL = bs_components_BL.copy()
        bs_components_BL[1] = -bs_components_BL[1]

        try:
            m = compute_peak_metrics(
                bs_rRTs_1, bs_rRTs_2,
                bs_Qbar_BL, bs_components_BL, bs_coords_BL,
                bs_Qbar_SD, bs_components_SD, bs_coords_SD,
                alphas, x_grid, x1_ind
            )
            all_bs_metrics.append(m)
        except Exception as e:
            print(f"  Skipping {fname}: {e}")

    # Aggregate
    keys = list(true_metrics.keys())
    bs_arr = {k: np.array([m[k] for m in all_bs_metrics if k in m]) for k in keys}

    for k in keys:
        print(f"  {k}: {np.nanmean(bs_arr[k]):.4f} ± {np.nanstd(bs_arr[k]):.4f}")

    output = {
        'true': {k: float(v) for k, v in true_metrics.items()},
        'bootstrap_mean': {k: float(np.nanmean(bs_arr[k])) for k in keys},
        'bootstrap_std':  {k: float(np.nanstd(bs_arr[k]))  for k in keys},
    }
    out_path = '../data/recon_peak_metrics_bootstrap.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
