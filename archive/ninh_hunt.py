from singleRegion import SingleRegionRecruitment
from multiRegion import MultiRegionRecruitment
import numpy as np
from scipy.stats import gamma, expon, geom
from scipy.optimize import minimize
from visualization import sim_multiHistogram
import math
from misc import *

## simulate N and n
def simulate_N_n(psi=1/50, gamma=5, n_trials=1, region_num=1):
    if region_num == 1:
        sim_N_n_data = []
        for _ in range(n_trials):
            sim_N = geom.rvs(p=psi)
            p_n_given_N = 1 / (gamma * sim_N + 1e-9)
            sim_n = geom.rvs(p=p_n_given_N)
            sim_N_n_data.append([sim_N, sim_n])
            
        return np.array(sim_N_n_data)
    else:
        return None


def loglik(alpha, beta, N, n, t):
    # Avoid division by zero and log of zero
    t = np.maximum(t, 1e-9) 
    
    theta = alpha / beta
    
    # Calculate the log-likelihood for each trial
    log_likelihood = (n - 1) * np.log(t) \
                     + theta * N * beta * np.log(beta) \
                     - (n + theta * beta * N) * np.log(t + beta) \
                     - log_beta(n, theta * beta * N)
    
    return np.sum(log_likelihood)


def neg_loglik(params, N, n, t):
    alpha, beta = params
    return -loglik(alpha, beta, N, n, t)

# For each simulation
alpha_true = 0.05
beta_true = 0.1
sim_N_n = simulate_N_n(psi=1/50, gamma=5, n_trials=100, region_num=1)

sim_N = sim_N_n[:, 0]
sim_n = sim_N_n[:, 1]
print("Simulated Data:")
print(f"N (first 5): {sim_N[:5]}")
print(f"n (first 5): {sim_n[:5]}")

recruitment_time = []
for N, n in sim_N_n:
    this_trial = SingleRegionRecruitment(n_trials=1, N=N, n=n, alpha=alpha_true, beta=beta_true)
    this_trial.genData()
    recruitment_time.append(this_trial.getRecruitmentTime())

sim_recruitment_times = np.array(recruitment_time).flatten()
print(f"t (first 5): {sim_recruitment_times[:5]}")

initial_guess = [0.1, 0.2]

# Perform the minimization
result = minimize(
    neg_loglik,
    initial_guess,
    args=(sim_N, sim_n, sim_recruitment_times),
    method='L-BFGS-B',
    bounds=[(1e-9, None), (1e-9, None)]
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