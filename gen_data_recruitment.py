from scipy.stats import gamma, expon, betaprime
import numpy as np
import matplotlib.pyplot as plt
import json, os

from visualization import sim_histogram

class MultiRegionRecruitment:
    def __init__(self, n_trials, M, N, n, alpha=None, beta=None, gamma_params=None):
        self.n_trials = n_trials 
        self.M = M                  # number of regions
        self.N = N                  # array, number of sites in each region
        self.n = n                  # number of patients to recruit
        self.gamma_params = gamma_params

        if gamma_params is not None:
            self._convert_params()
        else:
            self.alpha = alpha 
            self.beta = beta

        self._validate_params()

    def _validate_params(self):
        if len(self.M) != self.n_trials or len(self.N) != self.n_trials or len(self.n) != self.n_trials:
            raise ValueError("M, N, and n must have the same length as n_trials.")
        if self.alpha is not None and len(self.alpha) != self.n_trials:
            raise ValueError("alpha must have the same length as n_trials.")
        if self.beta is not None and len(self.beta) != self.n_trials:
            raise ValueError("beta must have the same length as n_trials.")
    
    def _convert_params(self):
        self.alpha = [[pair[0] for pair in trial] for trial in self.gamma_params.values()]
        self.beta = [[pair[1] for pair in trial] for trial in self.gamma_params.values()]
    
    def _genData_trial(self, trialIndex, seed=None):
        this_trial = {} # data structure for this particular trial

        # Get parameters for this trial
        this_M = self.M[trialIndex]             # no. regions in this trial
        this_N = self.N[trialIndex]             # no. sites in this trial    
        this_n = self.n[trialIndex]             # no. patients to recruit in this trial
        this_alpha = self.alpha[trialIndex]     # shape params for gamma distribution
        this_beta = self.beta[trialIndex]       # rate params for gamma distribution
        
        # Generate rates (flant array of rates for all sites)
        alphas = np.repeat(this_alpha, this_N)
        print(f"alphas: {alphas}")
        betas = np.repeat(this_beta, this_N)
        individual_rates = gamma.rvs(a=alphas, scale=1/betas, size=sum(this_N))

        count = 0
        recruitment_time = 0
        site_recruit_counts = np.zeros(len(individual_rates))
        # Init first arrival times
        arrival_times = expon.rvs(scale=1/individual_rates, size=len(individual_rates))

        while count < this_n:
            min_index = np.argmin(arrival_times)
            min_arrival_time = arrival_times[min_index]
            site_recruit_counts[min_index] += 1

            recruitment_time += min_arrival_time
            count += 1

            arrival_times -= min_arrival_time 
            arrival_times[min_index] = expon.rvs(scale=1/individual_rates[min_index], size=1)[0]

        # Init data structure for each region
        for i in range(this_M):
            this_trial[f"Region {i+1}"] = [0] * this_N[i]  # Initialize sites in each region
        
        # Fill in the recruitment counts for each site
        for i, count in enumerate(site_recruit_counts):
            region_index = next((j for j in range(this_M) if i < sum(self.N[trialIndex][:j+1])), None)
            if region_index is not None:
                this_trial[f"Region {region_index + 1}"][i - sum(self.N[trialIndex][:region_index])] = count
        
        this_trial["Total Recruitment Time"] = recruitment_time

        return this_trial

    def genData(self):
        self.data = {}
        for trialIndex in range(self.n_trials):
            self.data[f"Trial {trialIndex}"] = self._genData_trial(trialIndex)
        
        return self.data
    
    def saveData(self, filename):
        """ Save the generated data to a JSON file."""
        if not filename.endswith('.json'):
            filename += '.json'
        with open(filename, 'w') as f:
            json.dump(self.data, f, indent=4)
        
        full_path = os.path.abspath(filename)
        print(f"Data has been successfully saved to {full_path}")


# Example usage 1
n_trials = 2                                # number of simulations
M = [2, 5]                                  # number of regions
alpha = [[0.5, 2], 
         [0.5, 1, 0.75, 0.5, 2]]            # shape
beta = [[0.5, 4],
        [2, 2, 2, 0.5, 1]]                  # rate
N = [[2,3], [3,3,4,5,7]]                    # number of sites in each region
n = [50, 500]                               # number of patients to recruit

# test = MultiRegionRecruitment(n_trials, M, N, n, alpha, beta)
# data = test.genData()
# test.saveData('test1_recruitment_data.json')


# Example usage 2
n_trials = 2                                # number of simulations
M = [2, 5]                                  # number of regions
# (alpha, beta), each tuple represents (shape, rate) for each region
gamma_params = {"Trial 1": [(0.5, 2), (0.5, 4)],
                "Trial 2": [(0.5, 1), (0.75, 2), (0.5, 2), (2, 2), (1, 0.5)]} 

N = [[2,3], [3,3,4,5,7]]                    # number of sites in each region
n = [50, 500]                               # number of patients to recruit

test = MultiRegionRecruitment(n_trials, M, N, n, alpha, beta)
data = test.genData()
test.saveData('test2_recruitment_data.json')


 


