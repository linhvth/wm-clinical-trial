import numpy as np
from scipy.optimize import minimize
from scipy.stats import gamma
from misc import *
from singleRegion import SingleRegionRecruitment


sim_settings = [(10, 100), (50, 100), (100, 100), (1000, 100), (10000, 100)]  # (N, n)

alpha_true, beta_true = 2.0, 1.0
R = 200  # replicates

for N, n in sim_settings:
    alpha_hats, beta_hats = [], []
    for _ in range(R):
        trial = SingleRegionRecruitment(1, N, n, alpha=alpha_true, beta=beta_true)
        trial.genData()
        t_obs = np.array(trial.getRecruitmentTime())

        res = minimize(neg_loglik, x0=[1.0, 1.0],
                       args=(np.array([N]), np.array([n]), t_obs),
                       bounds=[(1e-9,None),(1e-9,None)])
        if res.success:
            ahat, bhat = res.x
            alpha_hats.append(ahat)
            beta_hats.append(bhat)

    print(f"N={N}, n={n}: "
          f"alpha_hat mean={np.mean(alpha_hats):.3f}, "
          f"beta_hat mean={np.mean(beta_hats):.3f}")
