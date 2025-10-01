import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from singleRegion import SingleRegionRecruitment
from misc import *

N_curr = 20
n_curr = 100
sim_settings = [5, 10, 20, 50, 100]  # number of trials in the past data
weights = [0.2, 0.5, 0.7, 0.9] # weight on past trials data

alpha_true, beta_true = 2.0, 1.0
R = 1000 # replicates
REQUIRED_COVERAGE = 80

# # --- File Saving Setup ---
# FIGURES_DIR = Path("figures")
# # Create the directory if it does not exist
# FIGURES_DIR.mkdir(exist_ok=True) 
# print(f"Saving figures to: {FIGURES_DIR.resolve()}")
# # -------------------------

for K in sim_settings:
    alpha_hats, beta_hats = [], []
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
                       args=(past_data, new_data, weights[1]),
                       method='L-BFGS-B', bounds=[(1e-9,None),(1e-9,None)])
        
        if res.success:
            ahat, bhat = res.x
            alpha_hats.append(ahat)
            beta_hats.append(bhat)

    alpha_hats = np.array(alpha_hats)
    beta_hats = np.array(beta_hats)
    
    # Check if there are successful runs before plotting
    if len(alpha_hats) == 0:
        print(f"No successful runs for K={K}. Skipping plot.")
        continue

    # Calculate the magnitude/ratio
    ratio_hats = alpha_hats / beta_hats
    
    # Print Summary Statistics
    print(f"K={K} (Successes: {len(alpha_hats)}/{R}):")
    print(f"  True: alpha={alpha_true}, beta={beta_true}, ratio={alpha_true/beta_true:.3f}")
    print(f"  Estimates: alpha_hat mean={np.mean(alpha_hats):.3f}, "
          f"beta_hat mean={np.mean(beta_hats):.3f}")
    print(f"  Ratio estimate mean={np.mean(ratio_hats):.3f}")

    # Plot Histograms
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.hist(alpha_hats, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(alpha_true, color='red', linestyle='dashed', linewidth=2)
    plt.title(f'Histogram of alpha estimates (K={K})')
    plt.xlabel('Estimated alpha')
    plt.ylabel('Frequency')

    plt.subplot