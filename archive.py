from scipy.stats import gamma, expon, betaprime
import numpy as np
import matplotlib.pyplot as plt

def sim_totalRate(n_sims, alpha, beta, N, n):
    """
    Simulate the time to recruit n patients from N sites using Monte Carlo simulation.
    This function generates individual recruitment rates from a Gamma distribution,
    sums them to get the total rate, and then samples from the Gamma distribution.
    """
    simulation_results = []
    for _ in range(n_sims):
        individual_rates = gamma.rvs(a=alpha, scale=1/beta, size=N)
        total_rate = np.sum(individual_rates)
        T_n_N = gamma.rvs(a=n, scale=1.0/total_rate)
        simulation_results.append(T_n_N)

    return simulation_results

def sim_minTimeRefresh(n_sims, N, n, alpha, beta):
    """
    Simulate the time to recruit n patients from N sites.
    Assumptions:
    - Recruitment rates follow a Gamma distribution.
    - Recruitment times are independent.
    """
    simulation_results = {"recruitment_time": [],
                          "recruitment_dist": []}
    
    for _ in range(n_sims):
        # Generate individual recruitment rates from a Gamma distribution
        individual_rates = gamma.rvs(a=alpha, scale=1/beta, size=N)

        site_recruit_counts = np.zeros(N)
        patient_counts = 0
        T_n_N = 0
        while patient_counts < n:
            # Sample recruitment time for each site
            time_waiting = expon.rvs(scale=1/individual_rates)

            # Find the site with the minimum recruitment time
            min_rate_index = np.argmin(time_waiting)
            
            # Update records
            patient_counts += 1
            site_recruit_counts[min_rate_index] += 1
            T_n_N += time_waiting[min_rate_index]

        simulation_results["recruitment_time"].append(T_n_N)
        simulation_results["recruitment_dist"].append(site_recruit_counts)
    
    return simulation_results

def sim_minTimeTrack(n_sims, N, n, alpha, beta):
    """
    Simulate the time to recruit n patients from N sites.
    Assumptions:
    - Recruitment rates follow a Gamma distribution.
    - Recruitment times are independent.
    """
    simulation_results = {"recruitment_time": [],
                          "recruitment_dist": []}
    
    for _ in range(n_sims):
        # Generate individual recruitment rates from a Gamma distribution
        individual_rates = gamma.rvs(a=alpha, scale=1/beta, size=N)
        arrival_times = expon.rvs(scale=1/individual_rates) #init

        site_recruit_counts = np.zeros(N)
        patient_counts = 0
        T_n_N = 0
        
        while patient_counts < n:
            # Find the site with the minimum recruitment time
            min_rate_index = np.argmin(arrival_times)
            
            # Update records
            patient_counts += 1
            site_recruit_counts[min_rate_index] += 1
            T_n_N += arrival_times[min_rate_index]

            # Update for the next iteration
            arrival_times -= arrival_times[min_rate_index]  # Decrease all times by the minimum
            arrival_times[min_rate_index] = expon.rvs(scale=1/individual_rates[min_rate_index])

        simulation_results["recruitment_time"].append(T_n_N)
        simulation_results["recruitment_dist"].append(site_recruit_counts)
    
    return simulation_results
