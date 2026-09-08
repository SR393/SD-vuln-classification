"""
Scatter plots of BL component coordinates vs SD component coordinates
with multivariate regression fits overlaid.

Recreates the diagonal plots from sd_bl_component_regression.py
with styling and figure size from lapse_rate_component_grid.py.
"""
import sys
sys.path.extend(['../data', '../analysis'])

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import json

import data_processing_funcs as dp


def plot_component_regression(ax, X, y, intercept, slope_x, slope_y_mean, 
                              X_mean, bl_comp_num, sd_comp_num, r_squared, 
                              uni_intercept=None, uni_slope=None, uni_r_squared=None,
                              labelsize=50):
    """
    Plot scatter of BL component vs SD component with regression fit.
    
    Args:
        ax: matplotlib axis
        X: (n,) BL component coordinates
        y: (n,) SD component coordinates
        intercept: scalar intercept of multivariate fit
        slope_x: scalar slope for this BL component
        slope_y_mean: scalar slope coefficient for other BL component
        X_mean: mean value of other BL component
        bl_comp_num: BL component number (1 or 2)
        sd_comp_num: SD component number (1 or 2)
        r_squared: R-squared value
        uni_intercept: univariate intercept for equation display
        uni_slope: univariate slope for equation display
        uni_r_squared: univariate R-squared for equation display
        labelsize: font size
    """
    s_point_size = 50
    
    # Scatter plot
    ax.scatter(X, y, alpha=0.6, s=s_point_size, color='steelblue', edgecolors='navy', linewidth=0.5, label = 'Individuals')
    
    # Regression line
    x_range = np.array([X.min(), X.max()])
    y_fit = intercept + slope_x * x_range + slope_y_mean * X_mean
    ax.plot(x_range, y_fit, 'r-', lw=4, label=f'Regression Line')
    
    # Add equation text with univariate results if provided
    if uni_intercept is not None and uni_slope is not None and uni_r_squared is not None:
        # Format equation: y = slope*x + intercept
        intercept_str = f"+ {uni_intercept:.3f}" if abs(uni_intercept) > 1e-10 else ""
        slope_str = f"{uni_slope:.3f}" if bl_comp_num == 1 else f"{uni_slope:.2f}"
        r2_str = f"{uni_r_squared:.3f}" if bl_comp_num == 1 else f"{uni_r_squared:.2f}"
        
        equation_text = f'$y = {slope_str}x {intercept_str}$\n$R^2 = {r2_str}$'
        
        # Place equation in upper right corner
        text_obj = ax.text(0.78, 0.575, equation_text, transform=ax.transAxes,
                          fontsize=32, verticalalignment='top', horizontalalignment='left',)
                          # bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Find point on regression line near the right end
        line_x = np.linspace(X.min(), X.max(), 100)
        line_y = uni_intercept + uni_slope * line_x
        connect_x = line_x[-10]
        connect_y = line_y[-10]
        
        # Draw connection patch from regression line to equation text using axes coordinates
        con = ConnectionPatch(
            xyA=(connect_x, connect_y), coordsA='data',
            xyB=(0.78, 0.575), coordsB='axes fraction',
            axesA=ax, axesB=ax,
            arrowstyle="-",
            linestyle='--',
            linewidth=1.5,
            color='black',
            alpha=0.6,
            zorder=1
        )
        ax.add_artist(con)
    
    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)
    
    # Set labels and title
    ax.set_xlabel(f'BL Component {bl_comp_num} Coordinate', fontsize=40, labelpad=10)
    ax.set_ylabel(f'SD Component {sd_comp_num} Coordinate', fontsize=40, labelpad=10)
    
    # Tick parameters
    ax.tick_params(axis='both', which='major', labelsize=35)


if __name__ == "__main__":

    K = 8

    # --- Load BL and SD components ---
    coords_BL = np.load(f'../data/logpca_coords_BL_{K}-dims.npy')
    coords_SD = np.load(f'../data/logpca_coords_SD_{K}-dims.npy')

    # Extract first 2 components
    X_BL = coords_BL[:, :2].copy()
    y_SD = coords_SD[:, :2].copy()

    # Apply BL component 2 sign flip (consistent with other analyses)
    X_BL[:, 1] = -X_BL[:, 1]

    print(f"N individuals: {X_BL.shape[0]}")
    print(f"BL components shape: {X_BL.shape}")
    print(f"SD components shape: {y_SD.shape}")

    # --- Load regression results ---
    with open('../data/sd_bl_component_regression.json', 'r') as f:
        results = json.load(f)
    
    mult = results['multivariate']
    intercept_mult = np.array(mult['intercept'])
    B_mult = np.array(mult['coefficient_matrix_B'])
    r_sq_mult = np.array(mult['r_squared_per_component'])

    # Load univariate results for equation display
    univariate = results['univariate']
    uni_c1_to_c1 = univariate['BL_c1_to_SD_c1']
    uni_c2_to_c2 = univariate['BL_c2_to_SD_c2']

    # --- Create plots ---
    fig, axes = plt.subplots(1, 2, figsize=(28, 14))

    labelsize = 70

    # Component 1: BL comp1 -> SD comp1
    plot_component_regression(
        axes[0], 
        X_BL[:, 0], y_SD[:, 0],
        intercept_mult[0], B_mult[0, 0], B_mult[1, 0], np.mean(X_BL[:, 1]),
        bl_comp_num=1, sd_comp_num=1, r_squared=r_sq_mult[0],
        uni_intercept=uni_c1_to_c1['intercept'],
        uni_slope=uni_c1_to_c1['slope'],
        uni_r_squared=uni_c1_to_c1['r_squared'],
        labelsize=labelsize
    )
    axis_1_ticks = np.linspace(-1.5, 1.5, 5)
    axes[0].text(-0.125, 1.01, 'A', fontsize=labelsize, transform=axes[0].transAxes, ha='right')
    axes[0].set_xticks(axis_1_ticks)
    axes[0].set_yticks(axis_1_ticks)
    axes[0].set_xlim([axis_1_ticks[0] - 0.05, axis_1_ticks[-1] + 0.05])
    axes[0].set_ylim([axis_1_ticks[0] - 0.05, axis_1_ticks[-1] + 0.05])
    axes[0].legend(fontsize=35, loc='upper left')

    # Component 2: BL comp2 -> SD comp2
    plot_component_regression(
        axes[1], 
        X_BL[:, 1], y_SD[:, 1],
        intercept_mult[1], B_mult[1, 1], B_mult[0, 1], np.mean(X_BL[:, 0]),
        bl_comp_num=2, sd_comp_num=2, r_squared=r_sq_mult[1],
        uni_intercept=uni_c2_to_c2['intercept'],
        uni_slope=uni_c2_to_c2['slope'],
        uni_r_squared=uni_c2_to_c2['r_squared'],
        labelsize=labelsize
    )
    axis_2_ticks = np.linspace(-0.5, 0.3, 5)
    axes[1].text(-0.1, 1.01, 'B', fontsize=labelsize, transform=axes[1].transAxes, ha='right')
    axes[1].set_xticks(np.linspace(-0.45, 0.3, 6))
    axes[1].set_yticks(axis_2_ticks)
    axes[1].set_xlim([-0.475, axis_2_ticks[-1] + 0.025])
    axes[1].set_ylim([axis_2_ticks[0] - 0.025, axis_2_ticks[-1] + 0.025])

    plt.tight_layout()
    plt.savefig('figure_3_regression.pdf', dpi=600)
