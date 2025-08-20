from scipy.stats import gamma, expon, betaprime
import numpy as np
import matplotlib.pyplot as plt
import json, os

from visualization import sim_histogram

class SingleRegionRecruitment:
    def __init__(self, n_trials, N, n, alpha=None, beta=None):
        self.n_trials = n_trials 
        self.N = N                  # number of sites
        self.n = n                  # number of patients to recruit
        self.alpha = alpha 
        self.beta = beta

    def genData(self, export_json=False, filename='single_region_data.json'):
        """
        Simulate the time to recruit n patients from N sites.
        Assumptions:
        - Recruitment rates follow a Gamma distribution.
        - Recruitment times are independent.
        """
        self.simulation_results = {"recruitment_time": [],
                                    "recruitment_dist": []}
        
        for _ in range(self.n_trials):
            # Generate individual recruitment rates from a Gamma distribution
            individual_rates = gamma.rvs(a=self.alpha, scale=1/self.beta, size=self.N)
            arrival_times = expon.rvs(scale=1/individual_rates) #init

            site_recruit_counts = np.zeros(self.N)
            patient_counts = 0
            T_n_N = 0
            
            while patient_counts < self.n:
                # Find the site with the minimum recruitment time
                min_rate_index = np.argmin(arrival_times)
                
                # Update records
                patient_counts += 1
                site_recruit_counts[min_rate_index] += 1
                T_n_N += arrival_times[min_rate_index]

                # Update for the next iteration
                arrival_times -= arrival_times[min_rate_index]  # Decrease all times by the minimum
                arrival_times[min_rate_index] = expon.rvs(scale=1/individual_rates[min_rate_index])

            self.simulation_results["recruitment_time"].append(T_n_N)
            self.simulation_results["recruitment_dist"].append(site_recruit_counts)

            if export_json:
                self._saveData(filename)
        
    def getRecruitmentTime(self):
        return self.simulation_results["recruitment_time"]
    
    def getRecruitmentDistribution(self):
        return self.simulation_results["recruitment_dist"]
    
    def _saveData(self, filename):
        """ Save the generated data to a JSON file."""
        if not filename.endswith('.json'):
            filename += '.json'
        with open(filename, 'w') as f:
            json.dump(self.data, f, indent=4)
        
        full_path = os.path.abspath(filename)
        print(f"Data has been successfully saved to {full_path}")