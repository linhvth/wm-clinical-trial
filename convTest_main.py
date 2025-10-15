from convTest_noSites import *
from convTest_noTrials import *

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pathlib import Path

### --- Global parameters ---
alpha_true, beta_true = 2.0, 1.0
R = 10 # replicates
REQUIRED_COVERAGE = 80

### --- No. Trials Convergence Test ---
N_curr = 20
n_curr = 100
sim_settings = [5, 20, 100]  # number of trials in the past data
weights = [0.2, 0.5, 0.7, 0.9] # weight on past trials data

# File Saving Setup
DATA_DIR = Path("data/noTrials")
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = Path("figures/noTrials")
FIGURES_DIR.mkdir(parents=True, exist_ok=True) 
print(f"Saving figures to: {FIGURES_DIR.resolve()}")

for K in sim_settings:
    w = weights[1]  # Using weight=0.5 for demonstration
    convTest_noTrials_sim(K, R, w, alpha_true, beta_true, N_curr, n_curr, DATA_DIR) # simulate and save results
    noTrials_extract_and_plot_results(K, R, w, alpha_true, beta_true, DATA_DIR, FIGURES_DIR) # plot from saved results
    

### --- No. Sites Convergence Test ---
sim_settings = [(10, 100), (50, 100), (100, 100), 
                (1000, 100), (10000, 100)]  # (N, n)

# File Saving Setup
DATA_DIR = Path("data/noSites")
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = Path("figures/noSites")
FIGURES_DIR.mkdir(parents=True, exist_ok=True) 
print(f"Saving figures to: {FIGURES_DIR.resolve()}")

for N, n in sim_settings:
    convTest_noSites_sim(N, n, R, alpha_true, beta_true, DATA_DIR) # simulate and save results
    noSites_extract_and_plot_results(N, n, R, alpha_true, beta_true, REQUIRED_COVERAGE, DATA_DIR, FIGURES_DIR) # plot from saved results