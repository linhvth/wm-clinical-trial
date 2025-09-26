import numpy as np
from scipy.stats import gamma, expon, geom
from scipy.optimize import minimize

import math, random, os
from misc import *
import pickle

from singleRegion import SingleRegionRecruitment
from visualization import sim_multiHistogram

random.seed(42)

data_file = 'simulated_data.pkl'

if os.path.exists(data_file):
    print("\n--- Loading Data from File ---")
    with open(data_file, 'rb') as f:
        loaded_data = pickle.load(f)
    
    sim_N_n = loaded_data['sim_N_n']
    sim_recruitment_times = loaded_data['sim_recruitment_times']
    N_new = loaded_data['N_new']
    n_new = loaded_data['n_new']
    new_recruitment_time = loaded_data['new_recruitment_time']
    new_recruitment_record = loaded_data['new_recruitment_record']
    alpha_true = loaded_data['alpha_true']
    beta_true = loaded_data['beta_true']
    alpha_est = loaded_data['alpha_est (past trials)']
    beta_est = loaded_data['beta_est (past trials)']

print(f"True alpha: {alpha_true}")
print(f"True beta: {beta_true}")

sim_N = sim_N_n[:, 0]
sim_n = sim_N_n[:, 1]

# --- Past trials information ---
print("\n--- Past Trials Information ---")
print(f"Estimate alpha from past trials: {alpha_est:.4f}")
print(f"Estimate beta from past trials: {beta_est:.4f}")

### Approach 1: Weighted MLE with new trial data
# Weighting factor for the new trial data  
weights = [0.2, 0.5, 0.7, 0.9] # weight on past trials data
mark_time = [0.1, 0.3, 0.5, 0.8] # mark times for interim data collection

# Combine past data with new trial data
past_data = {
    'N': sim_N,
    'n': sim_n,
    't': sim_recruitment_times
}

for t in mark_time:
    t = round(new_recruitment_time * t, 2)  # Interim time point
    n_t = sum(1 for record in new_recruitment_record if record["arrival_time"] <= t)

    # --- New trial information ---
    print("\n--- New Trial Parameters ---")
    print(f"New trial total sites N_new: {N_new}")
    print(f"New trial total patients n_new: {n_new}")
    print(f"New trial true recruitment time: {new_recruitment_time:.2f}")
    print(f"Interim data collected at time t={t:.2f}, number of patients recruited n(t)={n_t}")
    new_data = {
        'N': np.array([N_new]),
        'n': np.array([n_t]),
        't': np.array([t])
    }

    # Perform the minimization, Get starting points
    initial_guess = [0, 0]
    result = minimize(neg_loglik, initial_guess, 
                    args=(N_new, n_t, t),
                    method='L-BFGS-B', bounds=[(1e-9, None), (1e-9, None)]
    )
    print("\n--- Optimization Results (New Trial Only) ---")
    alpha_est_new, beta_est_new = result.x
    print(f"Estimated alpha (new trial only): {alpha_est_new:.4f}")
    print(f"Estimated beta (new trial only): {beta_est_new:.4f}")

    for w in weights:
        print(f"\nTime: {t}, Weighting Factor (Past trials): {w} ---")
        # Perform the minimization again
        result1 = minimize(neg_loglik_weighted, [0, 0], 
                        args=(past_data, new_data, w),
                        method='L-BFGS-B', bounds=[(1e-9, None), (1e-9, None)]
        )

        # Print the estimated parameters
        alpha_est1, beta_est1 = result1.x
        # print(f"Success: {result1.success}")
        # print(f"Message: {result1.message}")
        print(f"Estimated alpha (weighted MLE): {alpha_est1:.4f}")
        print(f"Estimated beta (weighted MLE): {beta_est1:.4f}")
        # print(f"Estimated t+: {(predict_remaining_time(alpha_est1, beta_est1, N_new, n_new-n_t)):.4f}")
        # print(f"True t+: {(new_recruitment_time - t):.4f}")
