from singleRegion import SingleRegionRecruitment
from scipy.optimize import minimize
from misc import *

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pathlib import Path
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

### --- Global parameters ---
alpha_true, beta_true = 2.0, 10.0
R = 50000 # replicates
K = 10 # number of trials in past data
global_t = 25.25 # observed time in current trial
N_curr = 20 # number of current sites
weights = [0, 0.25, 0.5, 0.75, 1] # weight on past trials data
# weights = [0, 0.5, 1] # quick test

# File Saving Setup
DATA_DIR = Path("data/homogeneity")
DATA_DIR.mkdir(parents=True, exist_ok=True)

FIGURES_DIR = Path("figures/homogeneity")
FIGURES_DIR.mkdir(parents=True, exist_ok=True) 
print(f"Saving figures to: {FIGURES_DIR.resolve()}")

RESULTS_DIR = Path("results/homogeneity")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ALPHA_RESULTS_FILE = RESULTS_DIR / "251025_50K_alpha_estimates.csv"
BETA_RESULTS_FILE = RESULTS_DIR / "251025_50K_beta_estimates.csv"

# Dynamically create keys for weights
w_keys = [f'w={w}' for w in weights]
all_run_keys = w_keys + ['combined'] # All keys we will store data in

# Dynamically initialize result dictionaries
alpha_hats = {key: [] for key in all_run_keys}
beta_hats = {key: [] for key in all_run_keys}

# Dynamically create plot labels and titles
plot_labels = [w for w in weights] + ['combined'] # Order for plotting
plot_titles = {}
for w in weights:
    if w == 0:
        plot_titles[w] = 'Weight = 0 (Current Only)'
    elif w == 1:
        plot_titles[w] = 'Weight = 1 (Past Only)'
    else:
        plot_titles[w] = f'Weight = {w}'
plot_titles['combined'] = 'Combined (Past + Current)'

### --- Main Simulation Loop ---
no_sites = [] # total no. sites in past trials
print(f"--- Starting {R} Replicates ---")
start_time = time.time() # <-- RECORD START TIME

for i in range(R): # for each replicate
    if (i+1) % 10000 == 0:
        curr_time =time.time() # <-- RECORD END TIME
        total_time = curr_time - start_time
        print(f"Running replicate {i+1}/{R}...")
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"({total_time / 60:.2f} minutes)")
        
    # Generate data once per replicate
    # Past trials data
    sim_N_n, t_obs= simulate_summary_data(alpha_true, beta_true, 
                                         psi=1/50, gamma=5, 
                                         n_trials=K, region_num=1)
    sim_N = sim_N_n[:, 0]
    sim_n = sim_N_n[:, 1]
    past_data = {'N': sim_N, 'n': sim_n, 't': t_obs}

    # Current trial data
    trial = SingleRegionRecruitment(1, N_curr, t=global_t, alpha=alpha_true, beta=beta_true)
    trial.genData()
    n_c_lst = trial.simulation_results["recruitment_dist"][0]
    new_data = {'N': np.array([N_curr]), 
                'n_c_lst': np.array(n_c_lst), 
                't': np.array([global_t])
                }
    
    # Run simulations for each weight setting
    for w in weights:
        # Solve for estimates
        res = minimize(neg_loglik_weighted, x0=[0.0, 0.0],
                       args=(past_data, new_data, w),
                       method='L-BFGS-B', bounds=[(1e-9,None),(1e-9,None)])
        
        # Store results
        if res.success:
            ahat, bhat = res.x
            alpha_hats[f'w={w}'].append(ahat)
            beta_hats[f'w={w}'].append(bhat)
    
    # Run the combined (unweighted) case
    res_comb = minimize(neg_loglik_combined, x0=[0.0, 0.0],
                        args=(past_data, new_data),
                        method='L-BFGS-B', bounds=[(1e-9,None),(1e-9,None)])
    if res_comb.success:
        ahat_c, bhat_c = res_comb.x
        alpha_hats['combined'].append(ahat_c)
        beta_hats['combined'].append(bhat_c)
        
    # Store total sites ONCE per replicate
    no_sites.append(np.sum(sim_N))

# Convert results to long-form DataFrames for easier plotting
alpha_df_list = []
for w_key, estimates in alpha_hats.items():
    # Handle 'w=X' keys and 'combined' key
    weight_label = w_key if w_key == 'combined' else float(w_key.split('=')[1])
    for est in estimates:
        alpha_df_list.append({'weight': weight_label, 'alpha_hat': est})
alpha_df = pd.DataFrame(alpha_df_list)

beta_df_list = []
for w_key, estimates in beta_hats.items():
    weight_label = w_key if w_key == 'combined' else float(w_key.split('=')[1])
    for est in estimates:
        beta_df_list.append({'weight': weight_label, 'beta_hat': est})
beta_df = pd.DataFrame(beta_df_list)

# --- Save results to disk ---
print(f"Saving estimate data to CSV files...")
try:
    alpha_df.to_csv(ALPHA_RESULTS_FILE, index=False)
    beta_df.to_csv(BETA_RESULTS_FILE, index=False)
    print(f"Successfully saved data to {RESULTS_DIR.resolve()}")
except Exception as e:
    print(f"Error saving data: {e}")
# --- End saving ---

print(f"Successfully saved grid plots to {FIGURES_DIR.resolve()}")

print("\n--- Quantitative Metrics (True Alpha=2.0, True Beta=1.0) ---")
print(f"{'Case':<10} | {'Alpha Bias':<12} | {'Alpha MSE':<12} | {'Beta Bias':<12} | {'Beta MSE':<12}")
print("-" * 64)

all_keys_for_table = [f'w={w}' for w in weights] + ['combined']

for w_key in all_keys_for_table:
    if w_key not in alpha_hats or not alpha_hats[w_key]:
        print(f"{w_key:<10} | {'N/A':<12} | {'N/A':<12} | {'N/A':<12} | {'N/A':<12}")
        continue
        
    a_hats = np.array(alpha_hats[w_key])
    b_hats = np.array(beta_hats[w_key])
    
    a_bias = np.mean(a_hats) - alpha_true
    b_bias = np.mean(b_hats) - beta_true
    
    a_mse = np.mean((a_hats - alpha_true)**2)
    b_mse = np.mean((b_hats - beta_true)**2)
    
    print(f"{w_key:<10} | {a_bias:<12.4f} | {a_mse:<12.4f} | {b_bias:<12.4f} | {b_mse:<12.4f}")