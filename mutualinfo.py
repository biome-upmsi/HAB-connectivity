# The manuscript reported Adjusted mutual info (AMI) values,
# but here, mutual information (MI) is also included as an alternative

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_mutual_info_score
import os

if not os.path.exists('output'):
    os.makedirs('output')


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
    #print('Mutual Info:', mutualinfo)
    return(mutualinfo)

'''
# TEST if implementation correct
X2 = [1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1]
X1 = [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1]
mi_12 = calc_MI(X2,X1,2)
print(mi_12)
#mi_12 = 0.00199 if log2(); 0.000599 if log10()
'''

df = pd.read_csv("input/bans_all.csv")       # v1 - irregular grid, updated to sep 2019
years = df.year.values

X = df.drop(['reference','datestring','year','month','datenumber'], axis='columns').values
#print(X[0:10, :])
#print(X.shape)

#only samples from year 1991-present
yearsub = years>=1991
X = X[yearsub, :]
print(X.shape)

# get unique values
xvals = np.unique(X)
nvals = len(xvals)
nsites = X.shape[1]
nr = X.shape[0]

MImat = np.zeros((nsites,nsites))
AMImat = np.zeros((nsites,nsites))

for s2 in range(0,nsites):
    for s1 in range(0,nsites):
        MImat[s2, s1] = calc_MI(X[:, s1], X[:, s2], nvals)
        AMImat[s2, s1] = adjusted_mutual_info_score(X[:, s1], X[:, s2])
        #if s1 != s2:
        #    MImat[s2, s1] = calc_MI(X[:, s1], X[:, s2], nvals)
        #    AMImat[s2, s1] = adjusted_mutual_info_score(X[:, s1], X[:, s2])

# Save matrix
np.savetxt("output/bans_AMI.csv", AMImat, delimiter=",")

var2plot = AMImat               # or MImat
varval = '$AMI(X,Y)$ (bits)'    # or '$MI(X,Y)$ (bits)'


fig, ax = plt.subplots()
plot2 = ax.imshow(var2plot,'gray_r')
ticks = range(0,11)
print(ticks)
ticklabels = ['Calbayog: 0','Cambatutay: 1','Irong-Irong: 2','Maqueda: 3','Villareal: 4','Daram Island: 5','Biliran: 6',
              'Carigara: 7','Coastal Leyte: 8','Calubian: 9','San Pedro Bay: 10']
ax.set_xticks(ticks)
#ax.set_xticklabels(ticklabels, rotation=70)
ax.set_yticks(ticks)
ax.set_yticklabels(ticklabels)
cbar = plt.colorbar(plot2)
cbar.set_label(varval, rotation=90)
#plt.tight_layout()
plt.show()
fig.savefig("output/fig_AMI.pdf", bbox_inches='tight')