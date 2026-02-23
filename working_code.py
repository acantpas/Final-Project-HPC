import filters
import pareto
import techniques
import histogram

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from ucimlrepo import fetch_ucirepo 


#################################### OBTAIN DATASET ####################################
########################################################################################
# fetch dataset 
glioma_grading_clinical_and_mutation_features = fetch_ucirepo(id=759) 
  
# data (as pandas dataframes) 
X = glioma_grading_clinical_and_mutation_features.data.features 
y = glioma_grading_clinical_and_mutation_features.data.targets 
 
 
races = X["Race"].unique()
X["Race"] = X["Race"].map({races[0]:0,races[1]:1,races[2]:2,races[3]:3})


###################################### FILTERS #########################################
########################################################################################
#filters.fscores(X,y)
#filters.relieff(X,y)

##################################### PARETO ###########################################
########################################################################################

X_selected = X[["IDH1", "Age_at_diagnosis", "PTEN", "ATRX", "CIC"]]

X_pca = pareto.Pareto(X_selected,5)
#print(techniques.NN(X_pca,y))

################################## TECHNIQUES ###########################################
#########################################################################################
acc_NN = []
pre_NN = []
acc_kMeans = []
pre_kMeans = []
acc_RF = []
pre_RF = []

for i in range(0,10):
    NN = techniques.NN(X_pca,y)
    acc_NN.append(NN[0])
    pre_NN.append(NN[1])

    kMeans = techniques.kMeans(X_pca,y,i)
    acc_kMeans.append(kMeans[0])
    pre_kMeans.append(kMeans[1])

    RF = techniques.randomForest(X_pca,y,i)
    acc_RF.append(RF[0])
    pre_RF.append(RF[1])


results = {
    'acc_NN': acc_NN, 'pre_NN': pre_NN,
    'acc_kMeans': acc_kMeans, 'pre_kMeans': pre_kMeans,
    'acc_RF': acc_RF, 'pre_RF': pre_RF
}

################################ DATA SAVE #########################################
####################################################################################
"""
# Data frame
df_results = pd.DataFrame(results)
df_results.to_csv('metrics.csv', index=False)
"""

############################### DATA LOAD ##########################################
####################################################################################
df = pd.read_csv('metrics.csv')
acc_NN = df['acc_NN'].tolist()
pre_NN = df['pre_NN'].tolist()

acc_kMeans = df['acc_kMeans'].tolist()
pre_kMeans = df['pre_kMeans'].tolist()

acc_RF = df['acc_RF'].tolist()
pre_RF = df['pre_RF'].tolist()

print(np.min(pre_NN))
print(np.max(pre_NN))
print(np.mean(pre_NN))
print(np.median(pre_NN))

################################# HISTOGRAMS #######################################
####################################################################################
"""
histogram.histogram(acc_NN,acc_kMeans,acc_RF,"Accuracy Score",'hist_acc.png')
histogram.histogram(pre_NN,pre_kMeans,pre_RF,"Precision Score",'hist_pre.png')
"""