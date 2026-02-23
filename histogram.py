import matplotlib.pyplot as plt
import numpy as np

################################# HISTOGRAM ################################
############################################################################
def histogram(NN, kMeans, RF, title, png):

    plt.figure(figsize=(10, 6))

    min_val = min(min(NN), min(RF))
    max_val = max(max(NN), max(RF))
    plt.hist(NN, bins=25, density=True, alpha=0.7, label='NN', color='skyblue',range=(min_val, max_val))
    plt.hist(RF, bins=25, density=True, alpha=0.7, label='Random Forest', color='lightgreen',range=(min_val, max_val))

    kMeans = np.mean(kMeans)
    plt.axvline(kMeans, color='salmon', linestyle='--', label=f'K-Means: {kMeans:.3f}')

    plt.title("Histogram values")
    plt.xlabel(title)
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(png)
