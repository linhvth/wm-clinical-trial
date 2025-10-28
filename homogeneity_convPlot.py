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
beta_true = 10.0
REQUIRED_COVERAGE = 80
R_total = 50000
R_check = 10000 

# Load the CSV files with error handling
try:
    # Using your original file paths
    df_alpha_full = pd.read_csv('results/homogeneity/251025_50k_alpha_estimates.csv')
    df_beta_full = pd.read_csv('results/homogeneity/251025_50k_beta_estimates.csv')
except FileNotFoundError:
    print("WARNING: Could not find 40k files, trying 'alpha_estimates.csv'")
    # Fallback to the names from the previous script
    df_alpha_full = pd.read_csv(Path('data/homogeneity') / 'alpha_estimates.csv')
    df_beta_full = pd.read_csv(Path('data/homogeneity') / 'beta_estimates.csv')


weights = df_alpha_full['weight'].unique()
print(f"Loaded data with weights: {weights}")

# Verify R_total
# Get the count of replicates for the first weight
r_loaded = len(df_alpha_full[df_alpha_full['weight'] == weights[0]])
if r_loaded != R_total:
    print(f"WARNING: R set to {R_total}, but CSV contains {r_loaded} replicates. Using {r_loaded} as R_total.")
    R_total = r_loaded
    if R_check > R_total:
        R_check = int(R_total * 0.6) # Adjust check to 60%
        print(f"Adjusting R_check to {R_check}")

print("Calculating running metrics... (this may take a moment)")

def get_running_metrics(df_full, true_value, param_name):
    """Calculates bias and MSE at intervals of R."""
    all_metrics = []
    step = 500
    r_steps = range(step, R_total + step, step)
    
    for w in df_full['weight'].unique():
        if w == 0.0:
            continue  # Skip w=0.0 to avoid scale issues
        # Get all data for this weight
        group_data = df_full[df_full['weight'] == w]
        
        for r in r_steps:
            # Get the first 'r' samples
            subset = group_data.head(r)
            
            # Calculate metrics for this subset
            bias = (subset[param_name] - true_value).mean()
            mse = ((subset[param_name] - true_value)**2).mean()
            
            all_metrics.append({
                'weight': w,
                'Replicates': r,
                'Bias': bias,
                'MSE': mse
            })
            
    return pd.DataFrame(all_metrics)

# Calculate for both alpha and beta
running_alpha_metrics = get_running_metrics(df_alpha_full, alpha_true, 'alpha_hat')
running_beta_metrics = get_running_metrics(df_beta_full, beta_true, 'beta_hat')

# Combine and Melt
running_alpha_metrics['Parameter'] = 'Alpha'
running_beta_metrics['Parameter'] = 'Beta'

# Melt each one
melted_alpha = running_alpha_metrics.melt(id_vars=['weight', 'Replicates', 'Parameter'], 
                                          value_vars=['Bias', 'MSE'], 
                                          var_name='Metric', value_name='Value')
melted_beta = running_beta_metrics.melt(id_vars=['weight', 'Replicates', 'Parameter'], 
                                        value_vars=['Bias', 'MSE'], 
                                        var_name='Metric', value_name='Value')

# Combine into one big DataFrame
all_running_metrics = pd.concat([melted_alpha, melted_beta])

# --- Plot ---
print("Plotting convergence traces...")

# Filter out the w=0.0 case, as its scale ruins the plot
remove1 = all_running_metrics['weight'] != '1.0'
remove2 = all_running_metrics['weight'] != '0.0'
# remove3 = all_running_metrics['weight'] != '0.75'
all_running_metrics_filtered = all_running_metrics[remove1 & remove2]

g = sns.relplot(
    data=all_running_metrics_filtered, # or all_running_metrics_filtered
    x='Replicates',
    y='Value',
    hue='weight',
    col='Metric',
    row='Parameter',
    kind='line',
    palette='Paired',
    linewidth=2,
    facet_kws={'sharey': False},
    legend=True
)

# Add horizontal lines at y=0 for Bias plots
for ax in g.axes_dict.values():
    ax_title = ax.get_title()
    if 'Bias' in ax_title:
        ax.axhline(0, ls='--', color='black', lw=1)

g.set_titles(col_template="{col_name}", row_template="{row_name}")
g.fig.suptitle("Convergence of Metrics (w=0.0 Excluded)", y=1.03)

# Add horizontal lines at y=0 for Bias plots
for ax in g.axes_dict.values():
    ax_title = ax.get_title()
    if 'Bias' in ax_title:
        ax.axhline(0, ls='--', color='black', lw=1)


# --- FIX 3: Move the *existing* seaborn legend ---

# Manually shrink the subplots to make room on the right
# This sets the right edge of the plots at 85% of the figure width
plt.subplots_adjust(right=0.85)

plot_path = FIGURES_DIR / "metrics_convergence_trace_filtered.png"
plt.savefig(plot_path) # Save the figure
print(f"Successfully saved improved filtered convergence plot to {plot_path}")
plt.close()
