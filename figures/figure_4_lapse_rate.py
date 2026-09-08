import sys
sys.path.extend(['../data', '../analysis'])

import numpy as np
import matplotlib.pyplot as plt

import data_processing_funcs as dp
from log_pca_analysis_funcs import reconstruct_quantiles

def predicted_lapse_rate(Q_recon, alphas, threshold=2.0):
    """
    Lapse rate from reconstructed quantile functions.
    P(rRT < threshold) = CDF(threshold) = alpha* s.t. Q(alpha*) = threshold.
    Uses interpolation after sorting Q to ensure monotonicity.
    """
    lapse_rates = np.empty(len(Q_recon))
    for i, Q in enumerate(Q_recon):
        sort_idx = np.argsort(Q)
        Q_sorted = Q[sort_idx]
        alphas_sorted = alphas[sort_idx]
        lapse_rates[i] = np.interp(
            threshold, Q_sorted, alphas_sorted,
            left=alphas_sorted[0], right=alphas_sorted[-1]
        )
    return lapse_rates

def lapse_rate_plot(ax, bins, pred_lapse_grid, c1_grid, N_C1, x_ticks, y_ticks, cols):

    s_point_size = 12
    labels = ["Component 2 Lower Tertile", "Component 2 Middle Tertile", "Component 2 Upper Tertile"]

    for bin_idx, bin_data in enumerate(bins):
        # Scatter individuals
        ax.scatter(bin_data['c1'], bin_data['lapse'], 
                   color=cols[bin_idx], alpha=0.9, s=s_point_size)
        
        # Plot predicted curve for this C2 bin
        curve_indices = np.arange(bin_idx * N_C1, (bin_idx + 1) * N_C1)
        pred_curve = pred_lapse_grid[curve_indices]
        ax.plot(c1_grid, pred_curve, color=cols[bin_idx], 
                linewidth=2, alpha=0.8, label=labels[bin_idx])

    ax.set_ylabel('Lapse Rate', fontsize=30)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.tick_params(axis='both', which='major', labelsize=35)

    ax.grid(True, alpha=0.3)
    ax.set_xlim([x_ticks[0] - 0.1, x_ticks[-1] + 0.1])
    ax.set_ylim([y_ticks[0] - 0.02, y_ticks[-1] + 0.02])

if __name__ == "__main__":

    # ===== USER PARAMETERS =====
    K = 8
    N_C1 = 40  # Number of component 1 values in grid
    N_C2 = 3   # Number of component 2 values (bin midpoints) in grid
    # ===========================

    eps = 1e-3
    M = 200
    alphas = np.linspace(eps, 1 - eps, M)
    da = alphas[1] - alphas[0]

    # --- Load log-PCA results ---
    coords_BL = np.load(f'../data/logpca_coords_BL_{K}-dims.npy')
    coords_SD = np.load(f'../data/logpca_coords_SD_{K}-dims.npy')

    components_BL = np.load(f'../data/logpca_components_BL_{K}-dims.npy')
    components_SD = np.load(f'../data/logpca_components_SD_{K}-dims.npy')

    mean_BL = np.load(f'../data/logpca_frechet_mean_BL_{K}-dims.npy')
    mean_SD = np.load(f'../data/logpca_frechet_mean_SD_{K}-dims.npy')

    # Apply BL component 2 sign flip
    coords_BL = coords_BL.copy()
    coords_BL[:, 1] = -coords_BL[:, 1]
    components_BL = components_BL.copy()
    components_BL[1] = -components_BL[1]

    # --- Load observed rRT data ---
    rRTs_1, rRTs_2 = dp.get_standard_individual_rRT_data()

    obs_lapse_count_BL = np.array([np.sum(rRT < 2.0) for rRT in rRTs_1])
    obs_lapse_count_SD = np.array([np.sum(rRT < 2.0) for rRT in rRTs_2])
    N_BL = np.array([len(rRT) for rRT in rRTs_1], dtype=float)
    N_SD = np.array([len(rRT) for rRT in rRTs_2], dtype=float)
    
    obs_lapse_BL = obs_lapse_count_BL / N_BL
    obs_lapse_SD = obs_lapse_count_SD / N_SD

    # --- Create component coordinate grids ---
    # C1 grid: evenly spaced
    # c1_grid_BL = np.linspace(coords_BL[:, 0].min(), coords_BL[:, 0].max(), N_C1)
    # c1_grid_SD = np.linspace(coords_SD[:, 0].min(), coords_SD[:, 0].max(), N_C1)
    c1_grid_BL = np.linspace(-1.4, 1.4, N_C1)
    c1_grid_SD = np.linspace(-1.4, 1.4, N_C1)
    
    # C2 grid: based on quantiles to create evenly-sized bins
    # Calculate quantile positions at (2*i+1)/(2*N_C2) for equal-sized groups
    c2_quantiles_BL = (2*np.arange(N_C2) + 1) / (2 * N_C2)
    c2_grid_BL = np.quantile(coords_BL[:, 1], c2_quantiles_BL)
    
    c2_quantiles_SD = (2*np.arange(N_C2) + 1) / (2 * N_C2)
    c2_grid_SD = np.quantile(coords_SD[:, 1], c2_quantiles_SD)

    # Define colors for each component 2 bin
    colors = plt.cm.viridis(np.linspace(0, 1, N_C2))


    print(f"Processing with N_C1={N_C1}, N_C2={N_C2}")

    # ===== Process BL data =====
    print("Processing BL data...")
    
    # Assign each individual to nearest C2 grid point
    c2_assignments_BL = np.zeros(len(coords_BL), dtype=int)
    for i in range(len(coords_BL)):
        c2_assignments_BL[i] = np.argmin(np.abs(coords_BL[i, 1] - c2_grid_BL))
    
    # Create bins for BL
    bins_BL = []
    for bin_idx in range(N_C2):
        mask = c2_assignments_BL == bin_idx
        if np.sum(mask) > 0:
            bins_BL.append({
                'c2_mid': c2_grid_BL[bin_idx],
                'c1': coords_BL[mask, 0],
                'lapse': obs_lapse_BL[mask],
                'color': colors[bin_idx],
                'n_individuals': np.sum(mask)
            })
    
    # Calculate predicted lapse rates on grid for BL
    grid_coords_BL = np.zeros((N_C1 * N_C2, 2))
    idx = 0
    for c2_idx in range(N_C2):
        for c1_idx in range(N_C1):
            grid_coords_BL[idx, 0] = c1_grid_BL[c1_idx]
            grid_coords_BL[idx, 1] = c2_grid_BL[c2_idx]
            idx += 1
    
    Q_recon_BL_grid = reconstruct_quantiles(mean_BL, components_BL[:2], grid_coords_BL[:, :2])
    pred_lapse_BL_grid = predicted_lapse_rate(Q_recon_BL_grid, alphas)

    # ===== Process SD data =====
    print("Processing SD data...")
    
    # Assign each individual to nearest C2 grid point
    c2_assignments_SD = np.zeros(len(coords_SD), dtype=int)
    for i in range(len(coords_SD)):
        c2_assignments_SD[i] = np.argmin(np.abs(coords_SD[i, 1] - c2_grid_SD))
    
    # Create bins for SD
    bins_SD = []
    for bin_idx in range(N_C2):
        mask = c2_assignments_SD == bin_idx
        if np.sum(mask) > 0:
            bins_SD.append({
                'c2_mid': c2_grid_SD[bin_idx],
                'c1': coords_SD[mask, 0],
                'lapse': obs_lapse_SD[mask],
                'color': colors[bin_idx],
                'n_individuals': np.sum(mask)
            })
    
    # Calculate predicted lapse rates on grid for SD
    grid_coords_SD = np.zeros((N_C1 * N_C2, 2))
    idx = 0
    for c2_idx in range(N_C2):
        for c1_idx in range(N_C1):
            grid_coords_SD[idx, 0] = c1_grid_SD[c1_idx]
            grid_coords_SD[idx, 1] = c2_grid_SD[c2_idx]
            idx += 1
    
    Q_recon_SD_grid = reconstruct_quantiles(mean_SD, components_SD[:2], grid_coords_SD[:, :2])
    pred_lapse_SD_grid = predicted_lapse_rate(Q_recon_SD_grid, alphas)

    # ===== Create plots =====
    fig, axes = plt.subplots(1, 2, figsize=(28, 14))

    labelsize = 70
    s_point_size = 15
    xticks = np.linspace(-1.5, 1.5, 5)
    yticks_BL = np.linspace(0, 0.8, 9)
    yticks_SD = np.linspace(0, 0.8, 9)

    # cols_BL = ['#d62728', '#9b4065', "#2c4ba0"]  # Example colors for BL
    # cols_BL = ['#482475', '#21918c', '#9bd93c']
    cols_BL = ['#41049d', '#cc4778', "#eaa11a"]
    # cols_SD = ['#d62728', '#9467bd', '#8c564b']  # Example colors for SD
    # cols_SD = ['#d62728', '#9b4065', "#2c4ba0"] 
    cols_SD = ['#41049d', '#cc4778', "#eaa11a"]
    
    # Plot BL
    lapse_rate_plot(axes[0], bins_BL, pred_lapse_BL_grid, c1_grid_BL, N_C1,
                    xticks, yticks_BL, cols_BL)
    axes[0].text(-0.165, 1.01, 'A', fontsize=labelsize, transform=axes[0].transAxes)
    axes[0].set_xlabel('BL Component 1 Coordinate', fontsize=40, labelpad=10)
    axes[0].set_ylabel('Lapse Rate', fontsize=40, labelpad=10)
    axes[0].legend(fontsize=30, loc='best')
    # Plot SD
    lapse_rate_plot(axes[1], bins_SD, pred_lapse_SD_grid, c1_grid_SD, N_C1,
                    xticks, yticks_SD, cols_SD)
    axes[1].text(-0.165, 1.01, 'B', fontsize=labelsize, transform=axes[1].transAxes)
    axes[1].set_xlabel('SD Component 1 Coordinate', fontsize=40, labelpad=10)
    axes[1].set_ylabel('')

    plt.savefig('figure_4_lapse_rate.pdf', dpi=600)
