from convTest_noSites import *
from convTest_noTrials import *

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pathlib import Path

### --- Global parameters ---
alpha_true, beta_true = 2.0, 10.0
R = 10000 # replicates
REQUIRED_COVERAGE = 80

### --- No. Trials Convergence Test ---
# expected t = 25.25
sim_settings = [1, 10, 20, 50, 100]  # number of trials in the past data

# # File Saving Setup
# DATA_DIR = Path("data/noTrials")
# DATA_DIR.mkdir(parents=True, exist_ok=True)
# FIGURES_DIR = Path("figures/noTrials")
# FIGURES_DIR.mkdir(parents=True, exist_ok=True) 
# print(f"Saving figures to: {FIGURES_DIR.resolve()}")

for K in sim_settings:
    convTest_noTrials_sim(K, R, alpha_true, beta_true) # simulate and save results
#     noTrials_extract_and_plot_results(K, R, w, alpha_true, beta_true, DATA_DIR, FIGURES_DIR) # plot from saved results
    

### --- No. Sites Convergence Test ---
sim_settings = [(10, 25.25), (50, 25.25), (100, 25.25), 
                (1000, 25.25), (10000, 25.25)]  # (N, t)

# # File Saving Setup
# DATA_DIR = Path("data/noSites")
# DATA_DIR.mkdir(parents=True, exist_ok=True)
# FIGURES_DIR = Path("figures/noSites")
# FIGURES_DIR.mkdir(parents=True, exist_ok=True) 
# print(f"Saving figures to: {FIGURES_DIR.resolve()}")

for N, t in sim_settings:
    convTest_noSites_sim(N, t, R, alpha_true, beta_true) # simulate and save results
    # noSites_extract_and_plot_results(N, t, R, alpha_true, beta_true, REQUIRED_COVERAGE, DATA_DIR, FIGURES_DIR) # plot from saved results