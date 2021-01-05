# The manuscript reported Adjusted mutual info (AMI) values,
# but here, mutual information (MI) is also included as an alternative

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_mutual_info_score


# Alternative option: Mutual information
def calc_MI(X1,X2,nvals):

    # init probabilities
    p_X1 = np.zeros(nvals)
    p_X2 = np.zeros(nvals)
    p_X1X2 = np.zeros((nvals, nvals))

    # get length of time series
    nt = len(X1)

    # calculate probabilities
    for i in range(0,nt):
        p_X1[X1[i]] += 1
        p_X2[X2[i]] += 1
        p_X1X2[X1[i],X2[i]] += 1
    p_X1 = p_X1 / sum(p_X1)
    p_X2 = p_X2 / sum(p_X2)
    p_X1X2 = p_X1X2 / sum(sum(p_X1X2))

    # calculate mutual information
    mutualinfo = 0.0
    for i in range(0,nvals):
        for j in range(0,nvals):
            p_xy = p_X1X2[i][j]
            p_x = p_X1[i]
            p_y = p_X2[j]
            if p_xy > 0.0 and (p_x * p_y) > 0.0:
                #Change base if needed; paper used log2
                mutualinfo += p_xy * math.log2(p_xy / (p_x * p_y))
    print('Mutual Info:', mutualinfo)
    return(mutualinfo)

'''
# TEST if implementation correct
X2 = [1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1]
X1 = [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1]
mi_12 = calc_MI(X2,X1,2)
print(mi_12)
#mi_12 = 0.00199 if log2(); 0.000599 if log10()
'''

df = pd.read_csv("bans_all.csv")       # v1 - irregular grid, updated to sep 2019
years = df.year.values

X = df.drop(['reference','datestring','year','month','datenumber'], axis='columns').values
#print(X[0:10, :])
#print(X.shape)

#only samples from year 1991-present
yearsub = years>=1991
X = X[yearsub, :]

