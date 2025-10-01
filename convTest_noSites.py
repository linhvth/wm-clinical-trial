import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from singleRegion import SingleRegionRecruitment
from misc import *

sim_settings = [(10, 100), (50, 100), (100, 100), 
                (1000, 100), (10000, 100)]  # (N, n)

alpha_true, beta_true = 2.0, 1.0
R = 1000 # replicates
REQUIRED_COVERAGE = 80

# --- File Saving Setup ---
FIGURES_DIR = Path("figures")
# Create the directory if it does not exist
FIGURES_DIR.mkdir(exist_ok=True) 
print(f"Saving figures to: {FIGURES_DIR.resolve()}")
# -------------------------

for N, n in sim_settings:
    alpha_hats, beta_hats = [], []
    for _ in range(R):
        trial = SingleRegionRecruitment(1, N, n, alpha=alpha_true, beta=beta_true)
        trial.genData()
        t_obs = np.array(trial.getRecruitmentTime())

        res = minimize(neg_loglik, x0=[1.0, 1.0],
                       args=(np.array([N]), np.array([n]), t_obs),
                       method='L-BFGS-B', bounds=[(1e-9,None),(1e-9,None)])
        if res.success:
            ahat, bhat = res.x
            alpha_hats.append(ahat)
            beta_hats.append(bhat)

    alpha_hats = np.array(alpha_hats)
    beta_hats = np.array(beta_hats)
    
    # Check if there are successful runs before plotting
    if len(alpha_hats) == 0:
        print(f"No successful runs for N={N}, n={n}. Skipping plot.")
        continue

    # Calculate the magnitude/ratio
    ratio_hats = alpha_hats / beta_hats
    
    # Print Summary Statistics
    print(f"N={N}, n={n} (Successes: {len(alpha_hats)}/{R}):")
    print(f"  True: alpha={alpha_true}, beta={beta_true}, ratio={alpha_true/beta_true:.3f}")
    print(f"  Estimates: alpha_hat mean={np.mean(alpha_hats):.3f}, "
          f"beta_hat mean={np.mean(beta_hats):.3f}")
    print(f"  Ratio estimate mean={np.mean(ratio_hats):.3f}")

    # Plot Histograms
    
    if len(alpha_hats) == 0:
        print(f"No successful runs for N={N}, n={n}. Skipping plot.")
        continue

    # Calculate the magnitude/ratio
    ratio_hats = alpha_hats / beta_hats

    # Calculate overall means
    alpha_mean = np.mean(alpha_hats)
    beta_mean = np.mean(beta_hats)

    # --- Dynamic Range Calculation to ensure >= 80% coverage ---
    
    def calculate_dynamic_zoom_range(estimates, mean, coverage_pct):
        """Calculates the [0, max_limit] range that meets the coverage requirement."""
        initial_max = 2.0 * mean
        
        # Calculate coverage for the initial range [0, initial_max]
        initial_count = np.sum(estimates <= initial_max)
        initial_coverage = (initial_count / len(estimates)) * 100
        
        if initial_coverage >= coverage_pct:
            # Case 1: 2*Mean already captures enough data
            final_max = initial_max
        else:
            # Case 2: Must extend the range to capture the required percentage
            # Use the percentile directly to get the required upper bound
            final_max = np.percentile(estimates, coverage_pct) * 1.05
        
        # Ensure the lower bound is 0
        final_min = 0.0 
        
        # Recalculate the final coverage for the chosen range (for reporting/title)
        final_count = np.sum((estimates >= final_min) & (estimates <= final_max))
        final_coverage = (final_count / len(estimates)) * 100
        
        return final_min, final_max, final_coverage

    # Apply the logic for Alpha
    alpha_plot_min, alpha_plot_max, alpha_percent_in_range = calculate_dynamic_zoom_range(
        alpha_hats, alpha_mean, REQUIRED_COVERAGE)

    # Apply the logic for Beta
    beta_plot_min, beta_plot_max, beta_percent_in_range = calculate_dynamic_zoom_range(
        beta_hats, beta_mean, REQUIRED_COVERAGE)


    # Print Summary Statistics 
    print(f"  Dynamic Zoom Ranges:")
    print(f"    alpha: [{alpha_plot_min:.3f}, {alpha_plot_max:.3f}] (Captures {alpha_percent_in_range:.2f}%)")
    print(f"    beta:  [{beta_plot_min:.3f}, {beta_plot_max:.3f}] (Captures {beta_percent_in_range:.2f}%) \n")


    # Plot Histograms
    ratio_hats = alpha_hats / beta_hats

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'MLE Estimates Distribution for $N={N}, n={n}$ ($R={len(alpha_hats)}$ successful runs)', fontsize=16)

    # Original Alpha Histogram (Wide Range)
    axes[0, 0].hist(alpha_hats, bins=30, density=True, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(alpha_true, color='red', linestyle='dashed', linewidth=1.5, label=f'True $\\alpha={alpha_true}$')
    axes[0, 0].axvline(alpha_mean, color='blue', linestyle='dotted', linewidth=1.5, label='Mean Estimate')
    axes[0, 0].set_title('Histogram of $\\hat{\\alpha}$ (Wide Range)')
    axes[0, 0].set_xlabel('$\\hat{\\alpha}$')
    axes[0, 0].legend(loc='upper right') 

    # Original Beta Histogram (Wide Range)
    axes[0, 1].hist(beta_hats, bins=30, density=True, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(beta_true, color='red', linestyle='dashed', linewidth=1.5, label=f'True $\\beta={beta_true}$')
    axes[0, 1].axvline(beta_mean, color='blue', linestyle='dotted', linewidth=1.5, label='Mean Estimate')
    axes[0, 1].set_title('Histogram of $\\hat{\\beta}$ (Wide Range)')
    axes[0, 1].set_xlabel('$\\hat{\\beta}$')
    axes[0, 1].legend(loc='upper right') 

    # Ratio Histogram
    ratio_true = alpha_true / beta_true
    axes[0, 2].hist(ratio_hats, bins=30, density=True, edgecolor='black', alpha=0.7)
    axes[0, 2].axvline(ratio_true, color='red', linestyle='dashed', linewidth=1.5, label=f'True Ratio={ratio_true:.3f}')
    axes[0, 2].axvline(np.mean(ratio_hats), color='blue', linestyle='dotted', linewidth=1.5, label='Mean Estimate')
    axes[0, 2].set_title('Histogram of $\\hat{\\alpha}/\\hat{\\beta}$')
    axes[0, 2].set_xlabel('$\\hat{\\alpha}/\\hat{\\beta}$')
    axes[0, 2].legend(loc='upper right') 


    # Zoomed Alpha Histogram (Dynamic Coverage)
    axes[1, 0].hist(alpha_hats, bins=30, range=(alpha_plot_min, alpha_plot_max), density=True, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(alpha_true, color='red', linestyle='dashed', linewidth=1.5, label=f'True $\\alpha={alpha_true}$')
    axes[1, 0].axvline(alpha_mean, color='blue', linestyle='dotted', linewidth=1.5, label='Overall Mean')
    axes[1, 0].set_title(f'Histogram of $\\hat{{\\alpha}}$ (Zoomed to [0, {alpha_plot_max:.2f}], {alpha_percent_in_range:.2f}% data)')
    axes[1, 0].set_xlabel('$\\hat{\\alpha}$')
    axes[1, 0].legend(loc='upper right', fontsize=8) 

    # Zoomed Beta Histogram (Dynamic Coverage)
    axes[1, 1].hist(beta_hats, bins=30, range=(beta_plot_min, beta_plot_max), density=True, edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(beta_true, color='red', linestyle='dashed', linewidth=1.5, label=f'True $\\beta={beta_true}$')
    axes[1, 1].axvline(beta_mean, color='blue', linestyle='dotted', linewidth=1.5, label='Overall Mean')
    axes[1, 1].set_title(f'Histogram of $\\hat{{\\beta}}$ (Zoomed to [0, {beta_plot_max:.2f}], {beta_percent_in_range:.2f}% data)')
    axes[1, 1].set_xlabel('$\\hat{\\beta}$')
    axes[1, 1].legend(loc='upper right', fontsize=8)

    # Hide the empty 6th subplot
    axes[1, 2].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    # Save the figure
    filename = FIGURES_DIR / f"Experiment_N{N}_n{n}_Dynamic80PctZoom.png"
    plt.savefig(filename)
    plt.close(fig)

    print(f"Saved figure to: {filename}")