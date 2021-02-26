import json
import numpy as np
import itertools
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler
from scipy import stats, linalg

#The partial correlation is derived from: https://gist.github.com/fabianp/9396204419c7b638d38f
"""
Partial Correlation in Python (clone of Matlab's partialcorr)
This uses the linear regression approach to compute the partial 
correlation (might be slow for a huge number of variables). The 
algorithm is detailed here:
    http://en.wikipedia.org/wiki/Partial_correlation#Using_linear_regression
Taking X and Y two variables of interest and Z the matrix with all the variable minus {X, Y},
the algorithm can be summarized as
    1) perform a normal linear least-squares regression with X as the target and Z as the predictor
    2) calculate the residuals in Step #1
    3) perform a normal linear least-squares regression with Y as the target and Z as the predictor
    4) calculate the residuals in Step #3
    5) calculate the correlation coefficient between the residuals from Steps #2 and #4; 
    The result is the partial correlation between X and Y while controlling for the effect of Z
Date: Nov 2014
Author: Fabian Pedregosa-Izquierdo, f@bianp.net
Testing: Valentina Borghesani, valentinaborghesani@gmail.com
"""

'''
def partial_corr(C):
    """
    Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling 
    for the remaining variables in C.
    Parameters
    ----------
    C : array-like, shape (n, p)
        Array with the different variables. Each column of C is taken as a variable
    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
        for the remaining variables in C.
    """
    C = np.asarray(C)
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i+1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0] #X,z X = NOC, Z = geselecteerde feats
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0] #y,z y = extra feat, z = 
            res_j = C[:, j] - C[:, idx].dot( beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)
            corr = stats.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr
    return P_corr
'''


# Getting the DATA
with open(r"D:\Documenten\TUdelft\thesis\mep_veldhuis\data\features5000\Features5000_278.txt") as json_file:
    FeaturesTraining = json.load(json_file)

training_ordered_features = [] 
training_labels = [] 

for sample, items in FeaturesTraining.items():
    sampleList = []
    featureNames = [] 
    training_labels.append(int(items['NOC']))
    for locus, item in sorted(items["Locus"].items()):
        for feature, featValue in sorted(item.items()):
            featureNames.append(feature)
            sampleList.append(float(featValue))
    del items['Locus']
    ordered = OrderedDict(sorted(items.items()))
    for key in sorted(items.keys()):
        featureNames.append(key)
    newSampleList = [*sampleList, *ordered.values()]
    training_ordered_features.append(list(map(float, newSampleList))) 

training_features = np.array(training_ordered_features)

# Normalise
scaler = StandardScaler()
training_features = scaler.fit_transform(training_features)

# Position of the NOC labels in featureNames
posNOC = [i for i,x in enumerate(featureNames) if x == "NOC"][0]

report  = open("partialCorrelationResultsTrain.txt", "w")
report.write("{0}\t{1}\t{2}\n".format("Feature", "AbsCorrelation", "vsFeats"))

#Correlation NOC vs feats
C = np.asarray(training_features)
p = C.shape[1]
P_corr = np.zeros((p, p), dtype=np.float)
NOC = C[:, posNOC]
posFeats = []    #Positions of selected feats
posFeats.append(posNOC)

# Set MAC as first feature
posMAC = [i for i,x in enumerate(featureNames) if x == "MAC"][0] # Position of MAC
correlations = []
for i in range(p):
    corr = stats.pearsonr(NOC, C[:, i])[0]
    if i in posFeats:
        corr = 0
    correlations.append(corr)
    print(featureNames[i], corr)

highestCorrelation = [i for i,x in enumerate(map(abs, correlations)) if x == max(map(abs, correlations))]
for high in highestCorrelation:
    print(featureNames[high])

print("Added feat: " + str(featureNames[posMAC]))
posFeats.append(posMAC)
report.write("{0}\t{1}\t{2}\n".format(str(featureNames[posMAC]), correlations[posMAC],list(np.array(featureNames)[posFeats])))

# Set TAC as second feature
posTAC = [i for i,x in enumerate(featureNames) if x == "TAC"][0] # Position of TAC
correlations = []
for i in range(p):
    corr = stats.pearsonr(NOC, C[:, i])[0]
    if i in posFeats:
        corr = 0
    correlations.append(corr)
    print(featureNames[i], corr)


highestCorrelation = [i for i,x in enumerate(map(abs, correlations)) if x == max(map(abs, correlations))]
for high in highestCorrelation:
    print(featureNames[high])

print("Added feat: " + str(featureNames[posTAC]))
posFeats.append(posTAC)
report.write("{0}\t{1}\t{2}\n".format(str(featureNames[posTAC]), correlations[posTAC],list(np.array(featureNames)[posFeats])))



# Partial correlation
for feat in range(p):
    correlationsPC = []
    for i in range(p):
        idx = np.zeros(p, dtype=np.bool)
        for pos in posFeats:
            if pos is not posNOC and pos is not i:
                idx[pos] = True
        beta_i = linalg.lstsq(C[:, idx], NOC)[0] #X,z X = NOC, Z = selected features
        beta_j = linalg.lstsq(C[:, idx], C[:, i])[0] #y,z y = extra feature, z = C[:, idx]
        res_j = NOC - C[:, idx].dot(beta_i)
        res_i = C[:, i] - C[:, idx].dot(beta_j)
        corr = stats.pearsonr(res_i, res_j)[0]
        if i in posFeats:
            corr = 0
        correlationsPC.append(corr)


    highestCorrelation = [i for i,x in enumerate(map(abs, correlationsPC)) if x == max(map(abs, correlationsPC))]
    if max(map(abs, correlationsPC)) == 0:
        continue
    for high in highestCorrelation:
        print(featureNames[high])

    posFeats.append(highestCorrelation[0])
    report.write("{0}\t{1}\t{2}\n".format(str(featureNames[highestCorrelation[0]]), max(map(abs, correlationsPC)), list(np.array(featureNames)[posFeats])))
    
report.close()