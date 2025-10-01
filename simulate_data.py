import numpy as np
from scipy.stats import gamma
from scipy.optimize import minimize

import math, random
from misc import *
import pickle

from singleRegion import SingleRegionRecruitment

random.seed(42)

K = 30                 # number of past trials to simulate
alpha_true = 0.05       # true alpha for Gamma distribution of recruitment rates
beta_true = 0.1         # true beta for Gamma distribution of recruitment rates
psi = 1/50              # parameter for Geometric distribution of number of sites
gamma = 5               # parameter for Geometric distribution of number of patients to recruit

### Simulate data for past trials, get alpha_est and beta_est by MLE
sim_N_n = helper_summary_simulate_N_n(psi=psi, gamma=gamma, n_trials=K, region_num=1)
sim_N = sim_N_n[:, 0]
sim_n = sim_N_n[:, 1]

# Enable to see the simulated N and n values separately
# print("Simulated Data:")
# print(f"Simulated (N, n) pairs (first 5): {sim_N_n[:5]}")
# print(f"N (first 5): {sim_N[:5]}")
# print(f"n (first 5): {sim_n[:5]}")

# Simulate recruitment times for each (N, n) pair
recruitment_time = []
for N, n in sim_N_n:
    this_trial = SingleRegionRecruitment(n_trials=1, N=N, n=n, 
                                         alpha=alpha_true, beta=beta_true)
    this_trial.genData()
    recruitment_time.append(this_trial.getRecruitmentTime())

sim_recruitment_times = np.array(recruitment_time).flatten()
# print(f"t (first 5): {sim_recruitment_times[:5]}")    # Enable to see the simulated recruitment times

# Perform the minimization, Get starting points
initial_guess = [0, 0]
result = minimize(neg_loglik, initial_guess, 
                  args=(sim_N, sim_n, sim_recruitment_times),
                  method='L-BFGS-B', bounds=[(1e-9, None), (1e-9, None)]
)

# Print the estimated parameters
alpha_est, beta_est = result.x
print("\n--- Optimization Results ---")
print(f"Success: {result.success}")
print(f"Message: {result.message}")
print(f"\nEstimated alpha: {alpha_est:.4f}")
print(f"Estimated beta: {beta_est:.4f}")
print(f"True alpha: {alpha_true}")
print(f"True beta: {beta_true}")

### New trial simulation
N_new, n_new  = simulate_N_n(psi=psi, gamma=gamma, n_trials=1, region_num=1)[0]
print("\n--- New Trial Parameters ---")
print(f"New Trial - N: {N_new}, n: {n_new}")

new_trial = SingleRegionRecruitment(n_trials=1, N=N_new, n=n_new, 
                                    alpha=alpha_true, beta=beta_true, 
                                    time_stamp=True)
new_trial.genData()
new_recruitment_time = new_trial.getRecruitmentTime()[0]
new_recruitment_dist = new_trial.getRecruitmentDistribution()[0]
new_recruitment_record = new_trial.getRecruitmentRecord()
print(f"New Trial - Recruitment Time: {new_recruitment_time:.4f}")

# Create a dictionary to hold all the data you want to save
data_to_save = {
    'sim_N_n': sim_N_n,
    'sim_recruitment_times': sim_recruitment_times,
    'N_new': N_new,
    'n_new': n_new,
    'new_recruitment_time': new_recruitment_time,
    'new_recruitment_record': new_recruitment_record,
    'alpha_true': alpha_true,
    'beta_true': beta_true,
    'alpha_est (past trials)': alpha_est,
    'beta_est (past trials)': beta_est
}

# Save the data to a file named 'simulated_data.pkl'
with open('simulated_data.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)

print("Data saved to 'simulated_data.pkl'.")