import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from singleRegion import SingleRegionRecruitment
from misc import *

def convTest_noTrials_sim(K, R, w, alpha_true, beta_true, N_curr, n_curr, DATA_DIR):
    alpha_hats, beta_hats = [], []
    no_sites = [] # total no. sites in past trials 
    for _ in range(R):
        # Past trials data
        sim_N_n, t_obs= simulate_summary_data(alpha_true, beta_true, psi=1/50, gamma=5, n_trials=K, region_num=1)
        sim_N = sim_N_n[:, 0]
        sim_n = sim_N_n[:, 1]
        past_data = {
                        'N': sim_N,
                        'n': sim_n,
                        't': t_obs
                    }

        # Current trial data
        trial = SingleRegionRecruitment(1, N_curr, n_curr, alpha=alpha_true, beta=beta_true)
        trial.genData()
        t_obs_new = np.array(trial.getRecruitmentTime())
        new_data = {
            'N': np.array([N_curr]),
            'n': np.array([n_curr]),
            't': np.array([t_obs_new])
        }

        res = minimize(neg_loglik_weighted, x0=[0.0, 0.0],
                       args=(past_data, new_data, w),
                       method='L-BFGS-B', bounds=[(1e-9,None),(1e-9,None)])
        
        if res.success:
            ahat, bhat = res.x
            alpha_hats.append(ahat)
            beta_hats.append(bhat)
            no_sites.append(np.sum(sim_N))

    alpha_hats = np.array(alpha_hats)
    beta_hats = np.array(beta_hats)
    no_sites = np.array(no_sites)

    # Calculate the magnitude/ratio
    ratio_hats = alpha_hats / beta_hats
    
    # Print Summary Statistics
    print(f"K={K} (Successes: {len(alpha_hats)}/{R}):")
    print(f"  True: alpha={alpha_true}, beta={beta_true}, ratio={alpha_true/beta_true:.3f}")
    print(f"  Estimates: alpha_hat mean={np.mean(alpha_hats):.3f}, "
          f"beta_hat mean={np.mean(beta_hats):.3f}")
    print(f"  Ratio estimate mean={np.mean(ratio_hats):.3f}")
    print(f"  Average no. sites in past trials: {np.mean(no_sites):.1f}")

    # Saving Results to File
    results_filename = DATA_DIR / f"sim_results_K={K}.npz"
    try:
        np.savez(
            str(results_filename),
            alpha_hats=alpha_hats,
            beta_hats=beta_hats,
            ratio_hats=ratio_hats,
            no_sites=no_sites
        )
        print(f"  Saved simulation results for K={K} to {results_filename}\n")
    except Exception as e:
        print(f"  Error saving data for K={K}: {e}")

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

    figure_filename = FIGURES_DIR / f"Experiment_K={K}.png"
    plt.savefig(figure_filename)
    plt.close(fig)
    print(f"  Saved histogram figure for K={K} to {figure_filename.name}")

def noTrials_extract_and_plot_results(K, R, current_weight, alpha_true, beta_true, DATA_DIR, FIGURES_DIR):
    """
    Loads saved simulation results and calls the plotting function.
    """
    results_filename = Path(f"{DATA_DIR}/sim_results_K={K}.npz")
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
