import math
import numpy as np
from scipy.special import loggamma

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