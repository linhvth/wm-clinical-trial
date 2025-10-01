plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.hist(alpha_hats, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(alpha_true, color='red', linestyle='dashed', linewidth=2)
    plt.title(f'Histogram of alpha estimates (K={K})')
    plt.xlabel('Estimated alpha')
    plt.ylabel('Frequency')

    plt.subplot