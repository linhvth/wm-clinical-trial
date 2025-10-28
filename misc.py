import math
import numpy as np
from scipy.special import loggamma, betaln
from scipy.stats import geom, betaprime
from singleRegion import SingleRegionRecruitment

def beta_function(x, y):
    if x <= 0 or y <= 0:
        raise ValueError("Arguments to the beta function must be positive.")
    
    # Calculate the gamma function for each argument
    gamma_x = math.gamma(x)
    gamma_y = math.gamma(y)
    gamma_x_plus_y = math.gamma(x + y)
    
    # Calculate the beta function
    beta_func = (gamma_x * gamma_y) / gamma_x_plus_y
    
    return beta_func

def log_beta(x, y):
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("Arguments to the log-beta function must be positive.")
    
    return loggamma(x) + loggamma(y) - loggamma(x + y)

def loglik(alpha, beta, N, n, t):
    """
    Calculate the log-likelihood of the observed recruitment times given parameters alpha and beta.
    N: array of number of sites for each trial
    n: array of number of patients to recruit for each trial
    t: array of observed recruitment times for each trial
    """
    # Avoid division by zero and log of zero
    t = np.maximum(t, 1e-9) 
    
    theta = alpha / beta
    
    # Calculate the log-likelihood for each trial
    log_likelihood = (n - 1) * np.log(t) \
                     + theta * N * beta * np.log(beta) \
                     - (n + theta * beta * N) * np.log(t + beta) \
                     - betaln(n, theta * beta * N)
    
    return np.sum(log_likelihood)

def neg_loglik(params, N, n, t):
    alpha, beta = params
    return -loglik(alpha, beta, N, n, t)

def mountain_loglik(alpha, beta, C, n_c_lst, t):
    """
    Calculate the log-likelihood of the observed recruitment times given parameters alpha and beta.
    C: number of sites/centers
    n_dot: total number of patients to recruit (realization of N.)
    t: censused observed recruitment times
    """
    n_c_lst = np.asarray(n_c_lst) # Ensure it's a numpy array
    t = np.maximum(t, 1e-9)

    # For alpha and beta very small, 0 * log(0) can occur. Assign 0 in these cases.
    term1 = np.where(alpha < 1e-30, 0.0, C * alpha * np.log(beta))

    sum_n_c = np.sum(n_c_lst)
    term2 = - (sum_n_c + C * alpha) * np.log(beta + t)

    # Avoid loggamma(0) when alpha is very small and n_i = 0
    n_c_positive = n_c_lst[n_c_lst > 0]
    num_positive_sites = len(n_c_positive)
    
    if num_positive_sites > 0:
        term3 = np.sum(loggamma(alpha + n_c_positive)) - num_positive_sites * loggamma(alpha)
    else: # If all sites had 0 patients
        term3 = 0.0
        
    # Total Log-Likelihood
    loglik = term1 + term2 + term3
    
    return loglik

def neg_loglik_mountain(params, C, n_c_lst, t):
    alpha, beta = params
    return -mountain_loglik(alpha, beta, C, n_c_lst, t)

def helper_summary_simulate_N_n(psi=1/50, gamma=5, n_trials=1, region_num=1):
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
    
def simulate_summary_data(alpha_true, beta_true, psi=1/50, gamma=5, n_trials=1, region_num=1):
    if region_num == 1:
        sim_N_n = helper_summary_simulate_N_n(psi=psi, gamma=gamma, 
                                                   n_trials=n_trials, region_num=region_num)

        # Simulate recruitment times for each (N, n) pair
        recruitment_time = []
        for N, n in sim_N_n:
            this_trial = SingleRegionRecruitment(n_trials=1, N=N, n=n, 
                                                alpha=alpha_true, beta=beta_true)
            this_trial.genData()
            recruitment_time.append(this_trial.getRecruitmentTime())

        sim_recruitment_times = np.array(recruitment_time).flatten()
        return sim_N_n, sim_recruitment_times
    else:
        return None, None # latter for multiregion case
    
def neg_loglik_weighted(params, past_data, new_data, gamma_weight):
    """
    Calculates the weighted negative log-likelihood for weighted MLE.
    
    Args:
        params (tuple): The parameters to be optimized (alpha, beta).
        past_data (dict): Data from past trials. Must contain 'N', 'n', 't'.
        new_data (dict): Data from the new trial. Must contain 'N', 'n', 't'.
        gamma_weight (float): The weight for past trials (0 to 1).
    """
    alpha, beta = params

    # Constraint check to ensure parameters are positive
    if alpha <= 0 or beta <= 0:
        return np.inf

    # Calculate log-likelihood for past trials
    loglik_past = loglik(
        alpha=alpha,
        beta=beta,
        N=past_data['N'],
        n=past_data['n'],
        t=past_data['t']
    )

    # Calculate log-likelihood for the new trial
    loglik_new = mountain_loglik(
        alpha=alpha,
        beta=beta,
        C=new_data['N'],
        n_c_lst=new_data['n_c_lst'],
        t=new_data['t']
    )

    # Combine with weights and return negative for minimization
    weighted_loglik = gamma_weight * loglik_past + (1 - gamma_weight) * loglik_new
    
    return -weighted_loglik

def neg_loglik_combined(params, past_data, new_data):
    """
    Calculates the combined negative log-likelihood (unweighted sum).
    This is equivalent to w=0.5, as minimizing -(A+B) is the same
    as minimizing -0.5*(A+B).
    """
    alpha, beta = params

    # Constraint check
    if alpha <= 0 or beta <= 0:
        return np.inf

    # Calculate log-likelihood for past trials
    loglik_past = loglik(
        alpha=alpha,
        beta=beta,
        N=past_data['N'],
        n=past_data['n'],
        t=past_data['t']
    )

    # Calculate log-likelihood for the new trial
    loglik_new = mountain_loglik(
        alpha=alpha,
        beta=beta,
        C=new_data['N'],
        n_c_lst=new_data['n_c_lst'],
        t=new_data['t']
    )
    
    # Return the simple negative sum
    return -(loglik_past + loglik_new)

def predict_remaining_time(alpha_est, beta_est, N, n):
    """
    Predicts the time to recruit the remaining patients using
    the Beta-Prime distribution.
    
    Args:
        alpha_est (float): The estimated alpha parameter.
        beta_est (float): The estimated beta parameter.
        N (int): The number of sites.
        n_total (int): The total number of patients to be recruited.
        n_current (int): The number of patients already recruited.

    Returns:
        float: The predicted time to recruit the remaining patients.
    """
    
    # Calculate the median of the Beta-Prime distribution
    predicted_t = betaprime.mean(a=n, b=alpha_est * N, scale=beta_est)
    
    return predicted_t

def calculate_dynamic_zoom_range(estimates, mean, coverage_pct):
    """Calculates the [0, max_limit] range that meets the coverage requirement."""
    initial_max = 2.0 * mean
        
    # Calculate coverage for the initial range [0, initial_max]
    initial_count = np.sum(estimates <= initial_max)
    initial_coverage = (initial_count / len(estimates)) * 100
        
    if initial_coverage >= coverage_pct:
            # Case 1: 2*Mean already captures enough data
        final_max = initial_max
    else:
        # Case 2: Must extend the range to capture the required percentage
        # Use the percentile directly to get the required upper bound
        final_max = np.percentile(estimates, coverage_pct) * 1.05
        
    # Ensure the lower bound is 0
    final_min = 0.0 
        
    # Recalculate the final coverage for the chosen range (for reporting/title)
    final_count = np.sum((estimates >= final_min) & (estimates <= final_max))
    final_coverage = (final_count / len(estimates)) * 100
        
    return final_min, final_max, final_coverage

def calculate_metrics(df, true_value, param_name):
    """Calculates bias and MSE for a given dataframe grouped by weight."""
    
    # Calculate Bias
    df['bias'] = df[param_name] - true_value
    
    # Calculate Squared Error
    df['se'] = (df[param_name] - true_value)**2
    
    # Group by weight and calculate mean bias and MSE
    metrics = df.groupby('weight').agg(
        Bias=('bias', 'mean'),
        MSE=('se', 'mean')
    ).reset_index()
    
    return metrics