# The manuscript reported Adjusted mutual info (AMI) values,
# but here, mutual information (MI) is also included as an alternative

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics.cluster import adjusted_mutual_info_score, mutual_info_score
import os
from randomdatasets import *


# Alternative option: Mutual information
def mutual_info(x1, x2):
    '''

    :param x1, x2: array, time series for site
    :return mutualinfo: mutual information of 2 sites
    '''
    # note that if x1=x2, MI(X,X) will be equal to entropy of (X)
    nvals = 3
    print("x1 = ", x1)

    # init probabilities
    p_X1 = np.zeros(nvals)
    p_X2 = np.zeros(nvals)
    p_X1X2 = np.zeros((nvals, nvals))

    # get length of time series
    nt = len(x1)

    # calculate probabilities
    for i in range(0, nt):
        p_X1[x1[i]] += 1
        p_X2[x2[i]] += 1
        p_X1X2[x1[i], x2[i]] += 1
    p_X1 = p_X1 / sum(p_X1)
    p_X2 = p_X2 / sum(p_X2)
    p_X1X2 = p_X1X2 / sum(sum(p_X1X2))

    # calculate mutual information
    mutualinfo = 0.0
    for i in range(0, nvals):
        for j in range(0, nvals):
            p_xy = p_X1X2[i][j]
            p_x = p_X1[i]
            p_y = p_X2[j]
            if p_xy > 0.0 and (p_x * p_y) > 0.0:
                #Change base if needed; paper used log2
                mutualinfo += p_xy * math.log2(p_xy / (p_x * p_y))
    #print('Mutual Info:', mutualinfo)
    return(mutualinfo)


def getPairwiseValues(func, X):
    '''

    :param func: function to calculate pairwise metric (options: adjusted_mutual_info_score or calc_MI)
    :param X: dataset, time x sites
    :return M: matrix of pairwise values, i destination, j source
    '''
    ntimepts, nsites = X.shape
    M = np.empty((nsites, nsites))
    M[:] = np.nan
    for i in range(0, nsites):
        for j in range(i, nsites):
            m = func(X[:, i], X[:, j])
            M[i, j] = m
            M[j, i] = m   # since symmetric measure
    return M


def getPairwiseRandStat(M3, func):
    '''

    :param M3: 3-dimensional array, replicate x sink site x source site
    :param func: statistic to calculate over M3[:, i, j]
    :return S: 2-dimensional array (nsites x nsites) of statistic
    '''
    nr, ni, nj = M3.shape
    S = np.empty((ni, ni))
    S[:] = np.nan

    for si in range(0, ni):
        for sj in range(si, nj):
            #print(M3[:, si, sj])
            S[si, sj] = func(M3[:, si, sj])
    return S


def plotPairwiseMutualInfo(X, cbar_text, filename=None, p = False, Xrand=0):
    '''

    :param X: matrix of pairwise values
    :param cbar_text: string, 'variable (units)' of M
    :param filename: to save plot
    :param p: bool, include percentile score in plot
    :param Xrand: a 3D array of random values of MI to use for percentile scoring
    :return: none
    '''
    #var2plot = randAMI[0, :, :]  # AMI
    #varval = '$AMI(X,Y)$ (bits)'  # or '$MI(X,Y)$ (bits)'

    fig, ax = plt.subplots()
    plot2 = ax.imshow(X, 'gray_r')
    ticks = range(0, 11)
    ticklabels = ['Calbayog: 0', 'Cambatutay: 1', 'Irong-Irong: 2', 'Maqueda: 3', 'Villareal: 4', 'Daram Island: 5',
                  'Biliran: 6',
                  'Carigara: 7', 'Coastal Leyte: 8', 'Calubian: 9', 'San Pedro Bay: 10']
    # Include percentile score based on random AMI values
    if p:
        for i in range(len(ticklabels)):
            for j in range(len(ticklabels)):
                if (i != j):
                    percentile = stats.percentileofscore(Xrand[:, i, j], X[i, j], 'weak')
                    if percentile > 95:
                        ax.text(j, i, round(percentile, 1), ha="center", va="center", color="royalblue", fontsize='x-small')
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticklabels)
    cbar = plt.colorbar(plot2)
    cbar.set_label(cbar_text, rotation=90)
    plt.show()
    if filename:
        fig.savefig(filename, bbox_inches='tight')




############ Output folder
if not os.path.exists('output'):
    os.makedirs('output')

############ Build training data
df = pd.read_csv("input/bans_all.csv")       # v1 - irregular grid, updated to sep 2019
years = df.year.values

X = df.drop(['reference', 'datestring', 'year', 'month', 'datenumber'], axis='columns').values
#print(X)

#only samples from year 1991-present
yearsub = years>=1991
X = X[yearsub, :]
nt, nsites = X.shape

############ Build random datasets
# Generate random datasets using Markov chains
nreps = 1000
randomDatasets = buildRandomDatasets(X, nreps)
print(randomDatasets[1, :, :])


############ Calculate AMI(i,j) for all i,j and datasets

# original dataset
AMI = getPairwiseValues(adjusted_mutual_info_score, X)
# Save values
np.savetxt("output/bans_AMI.csv", AMI, delimiter=",")
#plotPairwiseMutualInfo(AMI, '$AMI(X,Y)$ (bits)', "output/fig_AMI.pdf")

# random datasets
randAMI = np.empty((nreps, nsites, nsites))
randAMI[:] = np.nan
#randMI = np.empty((nreps, nsites, nsites))
#randMI[:] = np.nan

for r in range(0, nreps):
    randAMI[r, :, :] = getPairwiseValues(adjusted_mutual_info_score, randomDatasets[r, :, :])
    #randMI[r, :, :] = getPairwiseValues(mutual_info, randomDatasets[r, :, :])

#print(randAMI[0, :, :])
randAMImean = getPairwiseRandStat(randAMI, np.mean)
plotPairwiseMutualInfo(randAMImean, '$AMI(X,Y)$ (bits)', "output/fig_randAMImean.pdf")
#plotPairwiseMutualInfo(AMI, '$AMI(X,Y)$ (bits)', "output/fig_AMI_withpercentile.pdf", True, randAMI)



'''

############ Calculate probability of AMI[i,j] given the distribution of randAMI[:,i,j]
for i in range(0, nsites):
    for j in range(i, nsites):
        p = percentileofscore(randAMI[:, i, j], AMI[i, j], 'rank')
        print(i, ",", j, " = ", p)





# TEST if implementation correct
X2 = [1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1]
X1 = [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1]
mi_12 = mutual_info(X2,X1,2)
print(mi_12)
#mi_12 = 0.00199 if log2(); 0.000599 if log10()
'''

