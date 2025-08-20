from singleRegion import SingleRegionRecruitment
from multiRegion import MultiRegionRecruitment
import numpy as np
from scipy.stats import gamma, expon
from visualization import sim_multiHistogram

# Share params
n_trials = 10000

# Single Region case
n1 = 200
N1 = 10
alpha1 = 2.0
beta1 = 0.5
single_region_recruitment = SingleRegionRecruitment(n_trials, N=N1, n=n1, alpha=alpha1, beta=beta1)
singleReg_simulation_results = single_region_recruitment.genData()
singleReg_recruitment_time = single_region_recruitment.getRecruitmentTime()
singleReg_recruitment_dist = single_region_recruitment.getRecruitmentDistribution()

# Multi Region case
M2 = [2]*n_trials                                      # number of regions in each trial
alpha2 = [[2, 2] for _ in range(n_trials)]            # shape
beta2 = [[0.5, 0.5] for _ in range(n_trials)]          # rate
N2 = [[3,7] for _ in range(n_trials)]                   # number of sites in each region
n2 = [200]*n_trials                                     # number of patients to recruit
multi_region_recruitment = MultiRegionRecruitment(n_trials, M=M2, N=N2, n=n2, alpha=alpha2, beta=beta2)
multiReg_simulation_results = multi_region_recruitment.genData()
multiReg_recruitment_time = multi_region_recruitment.getRecruitmentTime()
multiReg_recruitment_dist = multi_region_recruitment.getRecruitmentDistribution()

# Visualization
sim_multiHistogram(
    singleReg_recruitment_time, multiReg_recruitment_time, N1, n1, alpha1, beta1, 
    label1='Single Region', label2='Multi Region',
    title='Compare')