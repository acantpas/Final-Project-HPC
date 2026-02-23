import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

############################################# K MEANS #######################################################
#############################################################################################################

def kMeans(X_vals, y, seed = 1):
    k = 2 
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    clusters = kmeans.fit_predict(X_vals)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    img = ax.scatter(X_vals[:, 0], X_vals[:, 1], X_vals[:, 2], c=clusters, cmap='viridis', alpha=0.6, s=40)
    
    ax.set_title(f'Patient clusters (K={k}) over 3D Space')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    
    fig.colorbar(img, ax=ax, label='Assigned cluster', shrink=0.5)
    plt.savefig('kmeans_3d.png')

    # Contingence table
    df_eval = pd.DataFrame({
        'Cluster_KMeans': clusters, 
        'Real': np.ravel(y) 
    })
    
    contingence = pd.crosstab(df_eval['Cluster_KMeans'], df_eval['Real'])
    print("\n--- Contingence table (Clusters vs Reality) ---")
    print(contingence)

    c0_r0 = contingence.iloc[0, 0]
    c0_r1 = contingence.iloc[0, 1]
    c1_r0 = contingence.iloc[1, 0]
    c1_r1 = contingence.iloc[1, 1]

    if (c0_r0 + c1_r1) < (c0_r1 + c1_r0):
        tp = c0_r1
        fp = c0_r0
        tn = c1_r0
        fn = c1_r1
    else:
        tp = c1_r1
        fp = c1_r0
        tn = c0_r0
        fn = c0_r1


    accuracy = float((tp + tn) / (tp + tn + fp + fn))
    precision = float(tp / (tp + fp))
    return([accuracy, precision])

############################################ RANDOM FOREST #######################################
##################################################################################################
def randomForest(X_selected,y, seed = 1):
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, np.ravel(y), test_size=0.20, random_state=seed)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=seed)
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)

    print("--- RANDOM FOREST ---")
    print(classification_report(y_test, y_pred, target_names=['LGG (0)', 'GBM (1)']))
    return([accuracy_score(y_test,y_pred),precision_score(y_test,y_pred)])

################################ NN ####################################
########################################################################
def NN(X_selected,y):
    X_t = torch.FloatTensor(StandardScaler().fit_transform(X_selected))
    y_t = torch.LongTensor(y.values.ravel())
    X_train, X_test, y_train, y_test = train_test_split(X_t, y_t, test_size=0.2)

    class NNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(X_train.shape[1], 128)
            self.fc2 = nn.Linear(128, 2)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            return F.log_softmax(self.fc2(x), dim=1)

    model = NNet()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    crit = nn.NLLLoss()

    for _ in range(50):
        opt.zero_grad()
        loss = crit(model(X_train), y_train)
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred_tensor = model(X_test).argmax(1)
        
        pred = model(X_test).argmax(1)
        acc = (pred == y_test).float().mean()
        print(f"Accuracy: {acc:.2%}")
        y_true_np = y_test.numpy()
        y_pred_np = pred_tensor.numpy()
        prec = float(precision_score(y_true_np, y_pred_np, average='binary'))
    return([float(acc),prec])
