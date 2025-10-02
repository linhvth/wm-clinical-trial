import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# --- File Saving Setup ---
FIGURES_DIR = Path("figures/noTrials")
# Create the directory if it does not exist
FIGURES_DIR.mkdir(exist_ok=True) 
print(f"Saving figures to: {FIGURES_DIR.resolve()}")
# -------------------------

def plot_histograms(K, alpha_hats, beta_hats, ratio_hats, no_sites, R, current_weight, alpha_true, beta_true, FIGURES_DIR):
    """
    Generates and saves a figure with four histograms for the estimation results.
    """
    ratio_true = alpha_true / beta_true
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Estimates Distribution for K={K} Past Trials (R={R}, Weight={current_weight})', 
                 fontsize=16, fontweight='bold')

    # Alpha Estimates
    ax = axes[0, 0]
    ax.hist(alpha_hats, bins=30, color='#6495ED', edgecolor='black', alpha=0.8)
    ax.axvline(alpha_true, color='red', linestyle='--', linewidth=2, label=f'True $\\alpha={alpha_true}$')
    mean_alpha_hat = np.mean(alpha_hats)
    ax.axvline(mean_alpha_hat, color='blue', linestyle='--', linewidth=2, label=f'Mean $\\hat{{\\alpha}}={mean_alpha_hat:.3f}$')
    ax.set_title(r'Histogram of $\hat{\alpha}$ Estimates', fontsize=12)
    ax.set_xlabel(r'Estimated $\alpha$')
    ax.set_ylabel('Percentage') 
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Beta Estimates
    ax = axes[0, 1]
    ax.hist(beta_hats, bins=30, color='#FF6347', edgecolor='black', alpha=0.8)
    ax.axvline(beta_true, color='red', linestyle='--', linewidth=2, label=f'True $\\beta={beta_true}$')
    mean_beta_hat = np.mean(beta_hats)
    ax.axvline(mean_beta_hat, color='blue', linestyle='--', linewidth=2, label=f'Mean $\\hat{{\\beta}}={mean_beta_hat:.3f}$')
    ax.set_title(r'Histogram of $\hat{\beta}$ Estimates', fontsize=12)
    ax.set_xlabel(r'Estimated $\beta$')
    ax.set_ylabel('Percentage') 
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Alpha/Beta Ratio Estimates
    ax = axes[1, 0]
    ax.hist(ratio_hats, bins=30, color='#3CB371', edgecolor='black', alpha=0.8)
    ax.axvline(ratio_true, color='red', linestyle='--', linewidth=2, label=f'True Ratio={ratio_true:.3f}')
    mean_ratio_hat = np.mean(ratio_hats)
    ax.axvline(mean_ratio_hat, color='blue', linestyle='--', linewidth=2, label=f'Mean Ratio={mean_ratio_hat:.3f}$')
    ax.set_title(r'Histogram of $\hat{\alpha}/\hat{\beta}$ Ratio', fontsize=12)
    ax.set_xlabel(r'Estimated Ratio ($\hat{\alpha}/\hat{\beta}$)')
    ax.set_ylabel('Percentage') 
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Number of Sites in Past Trials
    ax = axes[1, 1]
    ax.hist(no_sites, bins=30, color='#FFD700', edgecolor='black', alpha=0.8)
    ax.set_title('Histogram of Total Number of Sites in Past Trials', fontsize=12)
    ax.set_xlabel(r'Total Sites in Past Trials ($\sum N_{past}$)')
    ax.set_ylabel('Percentage')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    figure_filename = FIGURES_DIR / f"histograms_K_{K}.png"
    plt.savefig(figure_filename)
    plt.close(fig)
    print(f"  Saved histogram figure for K={K} to {figure_filename.name}")


def extract_and_plot_results(K, R, current_weight, alpha_true, beta_true, FIGURES_DIR):
    """
    Loads saved simulation results and calls the plotting function.
    """
    results_filename = Path(f"results_K_{K}.npz")
    if not results_filename.exists():
        print(f"File not found: {results_filename.name}. Cannot visualize.")
        return

    data = np.load(str(results_filename))
    
    alpha_hats = data['alpha_hats']
    beta_hats = data['beta_hats']
    ratio_hats = data['ratio_hats']
    no_sites = data['no_sites']
    
    if len(alpha_hats) == 0:
        print(f"No successful runs found in saved data for K={K}.")
        return

    ratio_true = alpha_true / beta_true
    print(f"\n--- Loaded results for K={K} (Successes: {len(alpha_hats)}/{R}) ---")
    print(f"  True: alpha={alpha_true}, beta={beta_true}, ratio={ratio_true:.3f}")
    print(f"  Estimates: alpha_hat mean={np.mean(alpha_hats):.3f}, "
          f"beta_hat mean={np.mean(beta_hats):.3f}")
    print(f"  Ratio estimate mean={np.mean(ratio_hats):.3f}")
    print(f"  Average no. sites in past trials: {np.mean(no_sites):.1f}")

    plot_histograms(K, alpha_hats, beta_hats, ratio_hats, no_sites, R, current_weight, 
                    alpha_true, beta_true, FIGURES_DIR)
    
    print(f"  Re-plotted figure for K={K} from saved data.")


for K in [5, 20, 100]:
    current_weight = 0.5
    extract_and_plot_results(K, R=100, current_weight=current_weight, 
                             alpha_true=2.0, beta_true=1.0, FIGURES_DIR=FIGURES_DIR)
