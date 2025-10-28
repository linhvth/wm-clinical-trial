from scipy.stats import gamma, expon, betaprime, geom
import numpy as np
import matplotlib.pyplot as plt
import json, os

from visualization import sim_histogram

class SingleRegionRecruitment:
    def __init__(self, n_trials, N, n=None, alpha=None, beta=None, t=None, time_stamp=False):
        self.n_trials = n_trials 
        self.N = N                  # number of sites
        self.n = n                  # number of patients to recruit
        self.alpha = alpha 
        self.beta = beta
        self.time_stamp = time_stamp
        self.t = t
    
    def helper_genData_given_noPatients(self):
        """
        Simulate the time to recruit n patients from N sites.
        Assumptions:
        - Recruitment rates follow a Gamma distribution.
        - Recruitment times are independent.
        """
        # if self.time_stamp:
        #     self.simulation_results["recruitment_record"] = []
        
        for _ in range(self.n_trials):
            # Generate individual recruitment rates from a Gamma distribution
            individual_rates = np.random.gamma(shape=self.alpha, scale=1/self.beta, size=self.N)
            arrival_times = np.random.exponential(scale=1/individual_rates)

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

                # if self.time_stamp:
                #     self.simulation_results["recruitment_record"].append({
                #         "patient_id": patient_counts,
                #         "site_id": int(min_rate_index) + 1,  # +1 for 1-based indexing
                #         "arrival_time": T_n_N
                #     })

                # Update for the next iteration
                arrival_times -= arrival_times[min_rate_index]  # Decrease all times by the minimum
                arrival_times[min_rate_index] = np.random.exponential(scale=1/individual_rates[min_rate_index])

            self.simulation_results["recruitment_time"].append(T_n_N)
            self.simulation_results["recruitment_dist"].append(site_recruit_counts)
            self.simulation_results["recruitment_patients"].append(patient_counts)
    
    def helper_genData_given_time_t(self):
        for _ in range(self.n_trials):
            # Generate individual recruitment rates from a Gamma distribution
            individual_rates = np.random.gamma(shape=self.alpha, scale=1/self.beta, size=self.N)
            arrival_times = np.random.exponential(scale=1/individual_rates)

            site_recruit_counts = np.zeros(self.N)
            patient_counts = 0
            count_time = 0
            
            while count_time < self.t:
                # Find the site with the minimum recruitment time
                min_rate_index = np.argmin(arrival_times)
                time_to_next_event = arrival_times[min_rate_index]

                if (count_time + time_to_next_event) > self.t:
                    break
                
                # Update records
                patient_counts += 1
                site_recruit_counts[min_rate_index] += 1
                count_time += time_to_next_event

                # Update for the next iteration
                arrival_times -= time_to_next_event # Decrease all times by the minimum
                arrival_times[min_rate_index] = np.random.exponential(scale=1/individual_rates[min_rate_index])

            self.simulation_results["recruitment_patients"].append(patient_counts)
            self.simulation_results["recruitment_dist"].append(site_recruit_counts)
            self.simulation_results["recruitment_time"].append(count_time)

    def genData(self, export_json=False, filename='single_region_data.json'):
        self.simulation_results = {"recruitment_time": [],
                                    "recruitment_dist": [],
                                    "recruitment_patients": [] }
        if self.t is not None:
            self.helper_genData_given_time_t()
        elif self.n is not None:
            self.helper_genData_given_noPatients()
        else:
            raise ValueError("Either number of patients 'n' or time 't' must be provided.")

        if export_json:
            self._saveData(filename)
        
    def getRecruitmentTime(self):
        return self.simulation_results["recruitment_time"]
    
    def getRecruitmentDistribution(self):
        return self.simulation_results["recruitment_dist"]
    
    def getRecruitmentRecord(self): 
        if self.time_stamp:
            return self.simulation_results["recruitment_record"]
        else:
            raise ValueError("Time stamp data was not recorded. Set time_stamp=True when initializing the class.")
    
    def _saveData(self, filename):
        """ Save the generated data to a JSON file."""
        if not filename.endswith('.json'):
            filename += '.json'
        with open(filename, 'w') as f:
            json.dump(self.data, f, indent=4)
        
        full_path = os.path.abspath(filename)
        print(f"Data has been successfully saved to {full_path}")