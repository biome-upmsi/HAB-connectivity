import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import os

if not os.path.exists('output'):
    os.makedirs('output')

def calc_TE(X1,X2,nvals,lag=1, pseucnt = 0.0):
    # Calculates the transfer entropy from X2 to X1
    t = len(X1)  # length of time series
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

    # calculate the transfer efficiency
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

'''
# TEST if implementation correct
X2 = [1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1]
X1 = [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1]
print(calc_TE(X1,X2,2,lag=1))
print(calc_TE(X2,X1,2,lag=1))
# 0.0440 or 0.0113 depending on which is X2 and X1; if using base 10 for the log
# 0.0375 or 0.1461 if base 2
'''

df = pd.read_csv("input/bans_monthly.csv")    # v2 - regular grid, some loss of resolution
X = df.drop(['reference','datestring','year','month','datenumber'], axis='columns').values


# get unique values
xvals = np.unique(X)
nvals = len(xvals)
nsites = X.shape[1]
nr = X.shape[0]

# Init array to hold TE given 1 month lag
Tmat = np.zeros((nsites,nsites))

# TE given lag of 1 month
for s2 in range(0,nsites):
    for s1 in range(0,nsites):
        if s1 != s2:
            # Calculate the transfer entropy from s2 to s1
            Tmat[s2, s1] = calc_TE(X[:,s1],X[:,s2],nvals,1,pseucnt=.0)
            # Note: row - source site; col - sink site
#np.savetxt("output/bans_TE_lag1.csv", Tmat, delimiter=",")


# With variable lags
maxlag = 4   # max lag in months
monthgrid = np.arange(1,maxlag+1)
#print(monthgrid)

# init arrays
TE = np.empty((nsites, nsites, maxlag))     # TE values per site pair given lag
TE[:] = np.NaN

# TE given lags of 1 to n months
for tlag in range(1, maxlag + 1):
    for s2 in range(0,nsites):          # iter over source
        for s1 in range(0,nsites):      # iter over sink
            if s1 != s2:
                TE[s2, s1, tlag - 1] = calc_TE(X[:, s1], X[:, s2], nvals, tlag, 0.0)
    filename = "output/bans_TE_given_lag_%i.csv" % tlag
    np.savetxt(filename, np.array(TE[:,:,tlag-1]), delimiter=",")
    # Note: row - source site; col - sink site
# Find the max TE across time lags
TEmax = np.amax(TE, axis=2)
np.savetxt("output/bans_TE_max.csv", TEmax, delimiter=",")
TEmaxlag = (TE.argmax(axis=2) + 1)
np.savetxt("output/bans_TE_lag_of_max.csv", TEmaxlag, delimiter=",")


# Plot TE: row - sink; col - source;
fig, ax1 = plt.subplots()
plot1 = ax1.imshow(Tmat.transpose(),'gray_r')
ticks = range(0,11)
ticklabels = ['Calbayog: 0','Cambatutay: 1','Irong-Irong: 2','Maqueda: 3','Villareal: 4','Daram Island: 5','Biliran: 6',
              'Carigara: 7','Coastal Leyte: 8','Calubian: 9','San Pedro Bay: 10']
ax1.set_xticks(ticks)
#ax1.set_xticklabels(ticklabels, rotation=90)
ax1.set_yticks(ticks)
ax1.set_yticklabels(ticklabels)
#plt.imshow(Tmat);
cbar = plt.colorbar(plot1)
cbar.set_label('$TE_{X \u2192 Y}$ | $\u03C4 = 1$ month (bits)', rotation=90)
#plt.tight_layout()
plt.show()
fig.savefig("output/fig_TE_lag_1_transpose.pdf", bbox_inches='tight')


# Plot TE max: row - sink; col - source;
# and include time lags for the 95th percentile values of all TEs
# set percentile value
cutval = np.percentile(TE[np.logical_not(np.isnan(TE))], 95)
#print('cutoff value: ', cutval)

TEmaxtrans = TEmax.transpose()          # x - source; y - destination
TEmaxlagtrans = TEmaxlag.transpose()
fig2, ax2 = plt.subplots()
plot2 = ax2.imshow(TEmaxtrans,'gray_r')
ticks = range(0,11)
# add the lags as text, but only for the high values
for i in range(len(ticklabels)):
    for j in range(len(ticklabels)):
        if TEmaxtrans[i,j] > cutval:
            text = ax2.text(j, i, int(TEmaxlagtrans[i,j]),
                       ha="center", va="center", color="w")


ticklabels = ['Calbayog: 0','Cambatutay: 1','Irong-Irong: 2','Maqueda: 3','Villareal: 4','Daram Island: 5','Biliran: 6',
              'Carigara: 7','Coastal Leyte: 8','Calubian: 9','San Pedro Bay: 10']
ax2.set_xticks(ticks)
#ax2.set_xticklabels(ticklabels, rotation=90)
ax2.set_yticks(ticks)
ax2.set_yticklabels(ticklabels)
cbar2 = plt.colorbar(plot2)
cbar2.set_label('max $TE_{X \u2192 Y}$ (bits)', rotation=90)
#plt.tight_layout()
plt.show()
fig2.savefig("output/fig_TE_max_transpose.pdf", bbox_inches='tight')