
from sklearn.feature_selection import f_classif
from skrebate import ReliefF

import pandas as pd


################################ FSCORES AND PVALUES ##############################
###################################################################################
def fscores(X,y):
    print("--- F SCORE ---")
    f_scores, p_values = f_classif(X, y.Grade)

    importancia = pd.DataFrame({
        'Variable': X.columns,
        'F_Score': f_scores,
        'P_Value': p_values
    })
    importancia = importancia.sort_values(by='F_Score', ascending=False)

    print(importancia)

def relieff(X,y):
################################### RELIEFF #############################################
#########################################################################################
    print("--- RELIEFF ---")
    fs = ReliefF(n_features_to_select=10, n_neighbors=100)

    fs.fit(X.values, y.Grade)

    importancias = fs.feature_importances_

    df_relief = pd.DataFrame({'Variable': X.columns, 'ReliefF_Score': importancias})
    df_relief = df_relief.sort_values(by='ReliefF_Score', ascending=False)
    print(df_relief)