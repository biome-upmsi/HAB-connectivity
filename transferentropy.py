import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats
from randomdatasets import *



def calc_TE(X1,X2,lag=1, pseucnt = 0.0):
    '''
    Calculates the transfer entropy from X2 to X1
    :param X1: destination time series
    :param X2: source time series
    :param nvals: number of states
    :param lag: lag time for X2, in months
    :param pseucnt: pseudo-count to add to the probability
    :return T_X2toX1: Transfer entropy from X2 to X1
    '''

    nvals = 3       # number of states
    t = len(X1)     # length of time series

    # calculate marginals
    # allowing for pseudo-counts
    p_X1 = np.ones(nvals) * pseucnt
    p_X1X2 = np.ones((nvals, nvals)) * pseucnt
    p_X1X1X2 = np.ones((nvals, nvals, nvals)) * pseucnt
    p_X1X1 = np.ones((nvals, nvals)) * pseucnt

    # sum counts of observed values
    for i in range(0, t):
        p_X1[X1[i]] += 1
        if (i >= lag):
            p_X1X1[X1[i], X1[i - 1]] += 1  # p(X1_n+1, X1_n)
            p_X1X2[X1[i - 1], X2[i - lag]] += 1
            p_X1X1X2[X1[i], X1[i - 1], X2[i - lag]] += 1  # p(X1_n+1, X1_n, X2_n-lag)
    # don't forget to add the last sample to the probability distribution of x1x2
    p_X1X2[X1[i], X2[i - (lag - 1)]] += 1

    # normalize to probabilities
    p_X1 = p_X1 / sum(p_X1)
    p_X1X2 = p_X1X2 / sum(sum(p_X1X2))
    p_X1X1 = p_X1X1 / sum(sum(p_X1X1))
    p_X1X1X2 = p_X1X1X2 / sum(sum(sum(p_X1X1X2)))

    # calculate the transfer entropy
    T_X2toX1 = 0.0
    '''
    for i in range(1,t):
        T_X2toX1 += p_X1X1X2[X1[i],X1[i-1],X2[i-1]] * math.log2(p_X1X1X2[X1[i],X1[i-1],X2[i-1]] * p_X1[X1[i-1]] /
                                                    (p_X1X1[X1[i],X1[i-1]]*p_X1X2[X1[i-1],X2[i-1]]))
    '''
    for i in range(0, nvals):  # iter over x1_n values
        for j in range(0, nvals):  # iter over x1_n-1 values
            for k in range(0, nvals):  # iter over x2_n-lag values
                pxxy = p_X1X1X2[i, j, k]
                px = p_X1[j]
                pxx = p_X1X1[i, j]
                pxy = p_X1X2[j, k]
                #print(pxxy)
                if pxxy * px > 0.0 and pxx * pxy > 0.0:
                    #T_X2toX1 += pxxy * math.log(pxxy * px / (pxx * pxy), 10)
                    T_X2toX1 += pxxy * math.log2(pxxy * px / (pxx * pxy))
    #print(T_X2toX1)
    return(T_X2toX1)

def buildNanArray(ni, nj, nk = None):
    if nk:
        arr = np.empty((nk, ni, nj))
    else:
        arr = np.empty((ni, nj))
    arr[:] = np.nan
    return arr

def getPairwiseTransferEntropy(X, lag, pseucnt=0, filename=None):
    '''
    Calculate transfer entropy for each site pair in the dataset
    :param X: matrix of time series, column - time, row - site
    :param lag: lag time for X2, in months
    :param pseucnt: pseudo-count to add to the probability
    :param filename: to save TE values
    :return Tmat: a matrix of pairwise TE values
    '''
    nt, nsites = X.shape
    Tmat = buildNanArray(nsites, nsites)

    for s2 in range(0, nsites):
        for s1 in range(0, nsites):
            if s1 != s2:
                # Calculate the transfer entropy from s2 to s1
                Tmat[s1, s2] = calc_TE(X[:, s1], X[:, s2], lag, pseucnt)
                # Note: row - destination site; col - source site
    if filename != None:
        np.savetxt(filename, Tmat, delimiter=",")
        #np.savetxt("output/bans_TE_lag1.csv", Tmat, delimiter=",")
    #print(np.nanmax(Tmat))
    return Tmat

def plotPairwiseTransferEntropy(X, cbar_text, filename=None, p = False, Xrand=0):
    # Plot TE: row - sink; col - source;
    fig, ax1 = plt.subplots()
    plot1 = ax1.imshow(X, 'gray_r', vmin=0, vmax=0.06)

    ticks = range(0, 11)
    ticklabels = ['Calbayog: 0', 'Cambatutay: 1', 'Irong-Irong: 2', 'Maqueda: 3', 'Villareal: 4', 'Daram Island: 5',
                  'Biliran: 6',
                  'Carigara: 7', 'Coastal Leyte: 8', 'Calubian: 9', 'San Pedro Bay: 10']

    for i in range(len(ticklabels)):
        for j in range(len(ticklabels)):
            if i == j:
                text = ax1.text(j, i, "-",
                                ha="center", va="center", color="k")

        # Include percentile score based on random AMI values
        if p:
            for i in range(len(ticklabels)):
                for j in range(len(ticklabels)):
                    if (i != j):
                        percentile = stats.percentileofscore(Xrand[:, i, j], X[i, j], 'weak')
                        if percentile > 95:
                            ax1.text(j, i, round(percentile, 1), ha="center", va="center", color="royalblue",
                                    fontsize='x-small')

    ax1.set_xticks(ticks)
    ax1.set_yticks(ticks)
    ax1.set_yticklabels(ticklabels)
    cbar = plt.colorbar(plot1)
    cbar.set_label(cbar_text, rotation=90)
    plt.show()
    if filename != None:
        fig.savefig(filename, bbox_inches='tight')


def getPairwiseStat(M3, func):
    '''
    :param M3: 3-dimensional array, k x sink site x source site
    :param func: statistic to calculate over M3[:, i, j]
    :return arrS: 2-dimensional array (nsites x nsites) of statistic
    '''
    nr, ni, nj = M3.shape
    arrS = buildNanArray(ni, ni)

    for si in range(0, ni):
        for sj in range(0, nj):
            #print(M3[:, si, sj])
            arrS[si, sj] = func(M3[:, si, sj])
    return arrS


############ Output folder
if not os.path.exists('output'):
    os.makedirs('output')


############ Build training data
df = pd.read_csv("input/bans_monthly.csv")    # v2 - regular grid, some loss of resolution
X = df.drop(['reference','datestring','year','month','datenumber'], axis='columns').values
nt, nsites = X.shape

# TE given variable lag months
maxlag = 4
TE_varlags = buildNanArray(nsites, nsites, maxlag)  # 3D array, pairwise TE at different lags
pseudocount = 0.0

for i in range(1, maxlag+1):
    print(i)
    tlag = i
    csvfilename = "output/bans_TE_lag%d.csv" %tlag
    Tmat = getPairwiseTransferEntropy(X, tlag, pseudocount, csvfilename)
    TE_varlags[i-1, :, :] = Tmat
    outputfigname = "output/fig_TE_lag_%d.pdf" % tlag
    if tlag == 1:
        colorbartitle = '$TE_{X \u2192 Y}$ | $\u03C4 = 1$ month (bits)'
    else:
        colorbartitle = '$TE_{X \u2192 Y}$ | $\u03C4 = %d$ months (bits)' % tlag
    plotPairwiseTransferEntropy(Tmat, colorbartitle, outputfigname, False)


maxTE = getPairwiseStat(TE_varlags, np.amax)        # max TE across 4 time lags
plotPairwiseTransferEntropy(maxTE, 'max $TE_{X \u2192 Y}$ (bits)',
                            "output/fig_TE_max.pdf", False)
np.savetxt("output/bans_TE_max.csv", maxTE, delimiter=",")



############ Build random datasets
# Generate random datasets using Markov chains
nreps = 1000
randomDatasets = buildRandomDatasets(X, nreps)

randTE = buildNanArray(nsites, nsites, nreps)       # TE values for each replicate dataset
randTEmax = buildNanArray(nsites, nsites, nreps)    # max TE values for each replicate dataset
for r in range(0, nreps):
    replicate_TE_4lags = buildNanArray(nsites, nsites, maxlag)
    for i in range(1, maxlag+1):
        TE = getPairwiseTransferEntropy(randomDatasets[r, :, :], i, pseudocount, None)
        replicate_TE_4lags[i - 1, :, :] = TE
        if i == 1:
            randTE[r, :, :] = TE
    randTEmax[r, :, :] = getPairwiseStat(replicate_TE_4lags, np.amax)

# mean TE given lag = 1 month across replicate datasets
randTEmean = getPairwiseStat(randTE, np.mean)
# plot mean TE given lag = 1 month across replicate datasets
plotPairwiseTransferEntropy(randTEmean,
                            'mean $TE_{X \u2192 Y}$ | $\u03C4 = 1$ month (bits) over random time series',
                            "output/fig_TErand_lag_1_mean.pdf", False)
# plot TE given 1 month lag from original data, with percentile score based on randTE
plotPairwiseTransferEntropy(TE_varlags[0, :, :],
                            '$TE_{X \u2192 Y}$ | $\u03C4 = 1$ month (bits)',
                            "output/fig_TE_lag_1_withpercentile.pdf",
                            True, randTE)

randTEmax_mean = getPairwiseStat(randTEmax, np.mean)
plotPairwiseTransferEntropy(randTEmax_mean,
                            'max $TE_{X \u2192 Y}$ (bits) averaged over random time series',
                            "output/fig_TErand_max_mean.pdf", False)

# plot max TE across 4 time lags from original data, with percentile score based on randTEmax
plotPairwiseTransferEntropy(maxTE, 'max $TE_{X \u2192 Y}$ (bits)',
                            "output/fig_TE_max_withpercentile.pdf", True, randTEmax)



'''
TEmax = np.amax(TE, axis=2)
np.savetxt("output/bans_TE_max.csv", TEmax, delimiter=",")
TEmaxlag = (TE.argmax(axis=2) + 1)
np.savetxt("output/bans_TE_lag_of_max.csv", TEmaxlag, delimiter=",")





# Plot TE max: row - sink; col - source;
# and include time lags for the 95th percentile values of all TEs
# set percentile value
cutval = np.percentile(TE[np.logical_not(np.isnan(TE))], 95)
#print('cutoff value: ', cutval)

TEmaxtrans = TEmax.transpose()          # x - source; y - destination
TEmaxlagtrans = TEmaxlag.transpose()
fig2, ax2 = plt.subplots()
plot2 = ax2.imshow(TEmaxtrans,'gray_r')

# add the lags as text, but only for the high values
for i in range(len(ticklabels)):
    for j in range(len(ticklabels)):
        if TEmaxtrans[i,j] > cutval:
            text = ax2.text(j, i, int(TEmaxlagtrans[i,j]),
                       ha="center", va="center", color="w")
        if i == j:
            text = ax2.text(j, i, "-",
                        ha="center", va="center", color="k")

ax2.set_xticks(ticks)
#ax2.set_xticklabels(ticklabels, rotation=90)
ax2.set_yticks(ticks)
ax2.set_yticklabels(ticklabels)
cbar2 = plt.colorbar(plot2)
cbar2.set_label('max $TE_{X \u2192 Y}$ (bits)', rotation=90)
#plt.tight_layout()
plt.show()
fig2.savefig("output/fig_TE_max_transpose.pdf", bbox_inches='tight')

'''



'''
# TEST if implementation correct
X2 = [1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1]
X1 = [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1]
print(calc_TE(X1,X2,2,lag=1))
print(calc_TE(X2,X1,2,lag=1))
# 0.0440 or 0.0113 depending on which is X2 and X1; if using base 10 for the log
# 0.0375 or 0.1461 if base 2
'''