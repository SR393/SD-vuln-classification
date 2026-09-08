import sys
sys.path.extend(['../data'])
import os
import json
from tqdm import tqdm

import numpy as np

import data_processing_funcs as dp
from log_pca_analysis_funcs import reconstruct_quantiles

def predicted_mean(Q_recon, da):
    return np.sum(Q_recon, axis=1) * da


def predicted_std(Q_recon, pred_mean, da):
    return np.sqrt(np.sum((Q_recon - pred_mean[:, None])**2, axis=1) * da)


def predicted_lapse_rate(Q_recon, alphas, threshold=2.0):
    lapse_rates = np.empty(len(Q_recon))
    for i, Q in enumerate(Q_recon):
        sort_idx = np.argsort(Q)
        lapse_rates[i] = np.interp(
            threshold, Q[sort_idx], alphas[sort_idx],
            left=alphas[sort_idx][0], right=alphas[sort_idx][-1]
        )
    return lapse_rates


def corr(a, b):
    return np.corrcoef(a, b)[0, 1]

def compute_metrics(rRTs_BL, rRTs_SD, Qbar_BL, components_BL, coords_BL,
                    Qbar_SD, components_SD, coords_SD, alphas, da, lapse_threshold=2.0):
    """ Returns a dict of {metric_name: value} for all three reconstructions
    (comp1, comp2, comp1+2) for both BL and SD."""

    # Observed
    obs_mean_BL = np.array([np.mean(r) for r in rRTs_BL])
    obs_mean_SD = np.array([np.mean(r) for r in rRTs_SD])
    obs_std_BL  = np.array([np.std(r)  for r in rRTs_BL])
    obs_std_SD  = np.array([np.std(r)  for r in rRTs_SD])
    obs_lapse_BL = np.array([np.mean(r < lapse_threshold) for r in rRTs_BL])
    obs_lapse_SD = np.array([np.mean(r < lapse_threshold) for r in rRTs_SD])
    n_obs_BL = np.array([len(r) for r in rRTs_BL])
    n_obs_SD = np.array([len(r) for r in rRTs_SD])

    # Reconstructions: comp1 only, comp2 only, comp1+2
    Q_BL_c1  = reconstruct_quantiles(Qbar_BL, components_BL[0:1], coords_BL[:, 0:1])
    Q_BL_c2  = reconstruct_quantiles(Qbar_BL, components_BL[1:2], coords_BL[:, 1:2])
    Q_BL_c12 = reconstruct_quantiles(Qbar_BL, components_BL[0:2], coords_BL[:, 0:2])

    Q_SD_c1  = reconstruct_quantiles(Qbar_SD, components_SD[0:1], coords_SD[:, 0:1])
    Q_SD_c2  = reconstruct_quantiles(Qbar_SD, components_SD[1:2], coords_SD[:, 1:2])
    Q_SD_c12 = reconstruct_quantiles(Qbar_SD, components_SD[0:2], coords_SD[:, 0:2])

    results = {}
    for cond, Q_c1, Q_c2, Q_c12, obs_mean, obs_std, obs_lapse, n_obs in [
        ('BL', Q_BL_c1, Q_BL_c2, Q_BL_c12, obs_mean_BL, obs_std_BL, obs_lapse_BL, n_obs_BL),
        ('SD', Q_SD_c1, Q_SD_c2, Q_SD_c12, obs_mean_SD, obs_std_SD, obs_lapse_SD, n_obs_SD),
    ]:
        for label, Q in [('c1', Q_c1), ('c2', Q_c2), ('c12', Q_c12)]:
            pred_mu    = predicted_mean(Q, da)
            pred_sigma = predicted_std(Q, pred_mu, da)
            pred_lapse = predicted_lapse_rate(Q, alphas, threshold=lapse_threshold)

            results[f'{cond}_{label}_r_mean']    = corr(obs_mean,  pred_mu)
            results[f'{cond}_{label}_r_std']     = corr(obs_std,   pred_sigma)
            results[f'{cond}_{label}_r_lapse']   = corr(obs_lapse, pred_lapse)

    return results

if __name__ == '__main__':

    K = 8
    LAPSE_THRESHOLD = 2.0
    eps = 1e-3
    M = 200
    alphas = np.linspace(eps, 1 - eps, M)
    da = alphas[1] - alphas[0]

    # Original data
    rRTs_1, rRTs_2 = dp.get_standard_individual_rRT_data()

    Qbar_BL      = np.load(f'../data/logpca_frechet_mean_BL_{K}-dims.npy')
    components_BL = np.load(f'../data/logpca_components_BL_{K}-dims.npy')
    coords_BL     = np.load(f'../data/logpca_coords_BL_{K}-dims.npy')

    Qbar_SD      = np.load(f'../data/logpca_frechet_mean_SD_{K}-dims.npy')
    components_SD = np.load(f'../data/logpca_components_SD_{K}-dims.npy')
    coords_SD     = np.load(f'../data/logpca_coords_SD_{K}-dims.npy')

    # Apply BL component 2 sign flip (consistent with other analyses)
    coords_BL = coords_BL.copy()
    coords_BL[:, 1] = -coords_BL[:, 1]
    components_BL = components_BL.copy()
    components_BL[1] = -components_BL[1]

    true_metrics = compute_metrics(
        rRTs_1, rRTs_2,
        Qbar_BL, components_BL, coords_BL,
        Qbar_SD, components_SD, coords_SD,
        alphas, da, lapse_threshold=LAPSE_THRESHOLD
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

        bs_Qbar_BL      = np.load(os.path.join(bs_dir, f'q_means/logpca_frechet_mean_BL_{K}-dims_{idx}.npy'))
        bs_components_BL = np.load(os.path.join(bs_dir, f'components/logpca_components_BL_{K}-dims_{idx}.npy'))
        bs_coords_BL     = np.load(os.path.join(bs_dir, f'coordinates/logpca_coords_BL_{K}-dims_{idx}.npy'))

        bs_Qbar_SD      = np.load(os.path.join(bs_dir, f'q_means/logpca_frechet_mean_SD_{K}-dims_{idx}.npy'))
        bs_components_SD = np.load(os.path.join(bs_dir, f'components/logpca_components_SD_{K}-dims_{idx}.npy'))
        bs_coords_SD     = np.load(os.path.join(bs_dir, f'coordinates/logpca_coords_SD_{K}-dims_{idx}.npy'))

        # Apply BL component 2 sign flip
        bs_coords_BL = bs_coords_BL.copy()
        bs_coords_BL[:, 1] = -bs_coords_BL[:, 1]
        bs_components_BL = bs_components_BL.copy()
        bs_components_BL[1] = -bs_components_BL[1]

        m = compute_metrics(
            bs_rRTs_1, bs_rRTs_2,
            bs_Qbar_BL, bs_components_BL, bs_coords_BL,
            bs_Qbar_SD, bs_components_SD, bs_coords_SD,
            alphas, da, lapse_threshold=LAPSE_THRESHOLD
        )
        all_bs_metrics.append(m)

    # Aggregate bootstrap statistics
    keys = list(true_metrics.keys())
    bs_arr = {k: np.array([m[k] for m in all_bs_metrics]) for k in keys}

    # Save
    output = {
        'true': {k: float(v) for k, v in true_metrics.items()},
        'bootstrap_mean': {k: float(np.nanmean(bs_arr[k])) for k in keys},
        'bootstrap_std':  {k: float(np.nanstd(bs_arr[k]))  for k in keys},
    }

    out_path = f'../data/recon_moment_metrics_bootstrap.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
