from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy as np

############################################# PARETO #####################################################
##########################################################################################################
def Pareto(X_selected,dimensions):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    pca = PCA(n_components=dimensions)
    X_pca = pca.fit_transform(X_scaled)
    pca.fit(X_scaled)

    exp_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(exp_var)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.bar(range(1, dimensions+1), exp_var, alpha=0.7, color='skyblue', label='Individual Variance')
    ax1.set_xlabel('Main component')
    ax1.set_ylabel('Explained Variance (Ratio)', color='blue')
    ax1.set_ylim(0, 1.0)

    ax2 = ax1.twinx()
    ax2.plot(range(1, dimensions+1), cum_var, marker='o', linestyle='-', color='red', label='Cumulative Variance')
    ax2.set_ylabel('Cumulative Variance (Total)', color='red')
    ax2.set_ylim(0, 1.0)

    # Reference lines
    ax2.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=0.9, color='gray', linestyle=':', alpha=0.5)

    plt.title('Pareto Chart')
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    plt.savefig('pareto_pca.png')
    return(X_pca)