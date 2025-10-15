for K in [5, 20, 100]:
    current_weight = 0.5
    extract_and_plot_results(K, R=100, current_weight=current_weight, 
                             alpha_true=2.0, beta_true=1.0, FIGURES_DIR=FIGURES_DIR)
