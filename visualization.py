from scipy.stats import gamma, expon, betaprime
import numpy as np
import matplotlib.pyplot as plt

def sim_histogram(simulation_results, N, n, alpha, beta, title=None):
    plt.figure(figsize=(10, 6))

    # Plot the histogram of the simulation results
    plt.hist(simulation_results, bins=50, density=True, 
            label='Monte Carlo Simulation Results', alpha=0.7)

    # Overlay the theoretical Beta Prime PDF
    # Parameters for Beta Prime: a=n, b=N*alpha, scale=beta
    x = np.linspace(min(simulation_results), max(simulation_results), 500)
    theoretical_pdf = betaprime.pdf(x, a=n, b=N*alpha, scale=beta)
    plt.plot(x, theoretical_pdf, 'r-', lw=2, 
            label=f'Theoretical Beta Prime PDF\n(n={n}, N={N}, alpha={alpha}. scale={beta})')

    plt.title(f'{title} - Distribution of Time to Recruit {n} Patients from {N} Sites')
    plt.xlabel('Time T(n,N)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def sim_multiHistogram(simulation_results1, simulation_results2, N, n, alpha, beta, 
                       title=None, label1='Simulation 1', label2='Simulation 2', color1='blue', color2='green'):
    plt.figure(figsize=(10, 6))

    # Plot the histogram of the first simulation results
    plt.hist(simulation_results1, bins=50, density=True,
             label=label1, alpha=0.6, color=color1)

    # Plot the histogram of the second simulation results
    plt.hist(simulation_results2, bins=50, density=True,
             label=label2, alpha=0.6, color=color2)

    # Overlay the theoretical Beta Prime PDF
    # Parameters for Beta Prime: a=n, b=N*alpha, scale=beta
    x = np.linspace(min(min(simulation_results1), min(simulation_results2)),
                    max(max(simulation_results1), max(simulation_results2)), 500)
    theoretical_pdf = betaprime.pdf(x, a=n, b=N*alpha, scale=beta)
    plt.plot(x, theoretical_pdf, 'r-', lw=2,
             label=f'Theoretical Beta Prime PDF\n(n={n}, N={N}, alpha={alpha}, scale={beta})')

    plt.title(f'{title} - Distribution of Time to Recruit {n} Patients from {N} Sites')
    plt.xlabel('Time T(n,N)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()