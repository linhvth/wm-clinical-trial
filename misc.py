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
    loglik_new = loglik(
        alpha=alpha,
        beta=beta,
        N=new_data['N'],
        n=new_data['n'],
        t=new_data['t']
    )

    # Combine with weights and return negative for minimization
    weighted_loglik = gamma_weight * loglik_past + (1 - gamma_weight) * loglik_new
    
    return -weighted_loglik

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