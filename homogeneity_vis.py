import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from misc import *
from pathlib import Path
import seaborn as sns

FIGURES_DIR = Path("figures/homogeneity")
FIGURES_DIR.mkdir(parents=True, exist_ok=True) 
print(f"Saving figures to: {FIGURES_DIR.resolve()}")

alpha_true = 2.0    
beta_true = 1.0
REQUIRED_COVERAGE = 80
R = 50000

# Load the CSV files
df1 = pd.read_csv('results/homogeneity/251025_50k_alpha_estimates.csv')
df2 = pd.read_csv('results/homogeneity/251025_50k_beta_estimates.csv')
weights = df1['weight'].unique()

# Separate alpha and beta data
alpha_df = df1[['weight', 'alpha_hat']]
beta_df = df2[['weight', 'beta_hat']]

# Dynamically create plot labels and titles
plot_labels = [w for w in weights] + ['combined'] # Order for plotting
plot_titles = {}
for w in weights:
    if w == 0:
        plot_titles[w] = 'Weight = 0 (Current Only)'
    elif w == 1:
        plot_titles[w] = 'Weight = 1 (Past Only)'
    else:
        plot_titles[w] = f'Weight = {w}'
plot_titles['combined'] = 'Combined (Past + Current)'

print("Plotting results...")

# --- Dynamic Plotting ---
num_plots = len(plot_labels)
ncols = 3 # Desired number of columns
nrows = int(np.ceil(num_plots / ncols)) # Calculate rows needed
figsize = (18, 6 * nrows) # Dynamic figure height

# Plot Alpha Estimates
fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True)
flat_axes = axes.flatten() # Flatten grid for easy iteration

for i, key in enumerate(plot_labels):
    ax = flat_axes[i]
    
    # Filter the data for the specific weight/case
    data_subset = alpha_df[alpha_df['weight'] == key]
    
    if data_subset.empty:
        ax.set_title(f'{plot_titles[key]} (No data)', fontsize=14)
        continue
        
    # Calculate dynamic zoom range
    all_estimates = data_subset['alpha_hat'].values
    mean_est = np.mean(all_estimates)
    xmin, xmax, coverage = calculate_dynamic_zoom_range(
        all_estimates, mean_est, REQUIRED_COVERAGE
    )
    
    # Plot histogram for that weight
    sns.histplot(data=data_subset, x='alpha_hat', kde=True, ax=ax, element="step")
    
    # Add the "true value" line
    ax.axvline(x=alpha_true, color='red', linestyle='--', linewidth=2, 
              label=f'True Alpha = {alpha_true}')
    
    # Apply zoom and set title
    ax.set_xlim(xmin, xmax)
    ax.set_title(f'{plot_titles[key]} ({coverage:.1f}% shown)', fontsize=14)
    ax.legend()

# Hide any unused subplots
for i in range(num_plots, len(flat_axes)):
    fig.delaxes(flat_axes[i])

# Add one big title and shared axis labels
fig.suptitle(f'Distribution of Alpha Estimates (R={R} replicates)', fontsize=18, y=1.02)
fig.supxlabel('Alpha Estimate', fontsize=14)
fig.supylabel('Density', fontsize=14)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'alpha_estimates_grid.png')
plt.close()


# Plot Beta Estimates
fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True)
flat_axes = axes.flatten() # Flatten grid for easy iteration

for i, key in enumerate(plot_labels):
    ax = flat_axes[i]
    
    # Filter the data for the specific weight
    data_subset = beta_df[beta_df['weight'] == key]
    
    if data_subset.empty:
        ax.set_title(f'{plot_titles[key]} (No data)', fontsize=14)
        continue
        
    # Calculate dynamic zoom range
    all_estimates = data_subset['beta_hat'].values
    mean_est = np.mean(all_estimates)
    xmin, xmax, coverage = calculate_dynamic_zoom_range(
        all_estimates, mean_est, REQUIRED_COVERAGE
    )
    
    # Plot histogram for that weight
    sns.histplot(data=data_subset, x='beta_hat', kde=True, ax=ax, element="step")
    
    # Add the "true value" line
    ax.axvline(x=beta_true, color='red', linestyle='--', linewidth=2, 
              label=f'True Beta = {beta_true}')
    
    # Apply zoom and set title
    ax.set_xlim(xmin, xmax)
    ax.set_title(f'{plot_titles[key]} ({coverage:.1f}% shown)', fontsize=14)
    ax.legend()

# Hide any unused subplots
for i in range(num_plots, len(flat_axes)):
    fig.delaxes(flat_axes[i])

# Add one big title and shared axis labels
fig.suptitle(f'Distribution of Beta Estimates (R={R} replicates)', fontsize=18, y=1.02)
fig.supxlabel('Beta Estimate', fontsize=14)
fig.supylabel('Density', fontsize=14)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'beta_estimates_grid.png')
plt.close()
# --- End Dynamic Plotting ---

print(f"Successfully saved grid plots to {FIGURES_DIR.resolve()}")
