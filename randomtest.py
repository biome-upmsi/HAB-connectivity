from pomegranate import *
import pandas as pd
import numpy as np
from transferentropy import calc_ami


def fitMarkovChain(X):
    d1 = DiscreteDistribution({0: 1. / 3, 1: 1. / 3, 2: 1. / 3})
    d2 = ConditionalProbabilityTable([[0, 0, 1. / 3],
                                      [0, 1, 1. / 3],
                                      [0, 2, 1. / 3],
                                      [1, 0, 1. / 3],
                                      [1, 1, 1. / 3],
                                      [1, 2, 1. / 3],
                                      [2, 0, 1. / 3],
                                      [2, 1, 1. / 3],
                                      [2, 2, 1. / 3],
                                      ], [d1])
    mc = MarkovChain([d1, d2])
    mc.fit([X])
    #print(mc.distributions)
    return mc

def buildRandomDatasets(X, nr):
    '''
    Iterate over each discrete variable in the matrix X
        Fit a Markov Chain
        Generate random samples
    :param X: time(row) x site(col) bans data
    :param nr: number of random datasets to build
    :return: 3dim array time x site x nr
    '''
    ntime,nsites = X.shape
    Xrands =  np.empty((nr, ntime, nsites))
    Xrands[:] = np.nan
    #print(Xrands[0,:,:].shape)
    for s in range(0,nsites):
        mc = fitMarkovChain(X[:,s])
        for r in range(0,nr):
            Xrands[r, :, s] = mc.sample(ntime)



############ Build training data
df = pd.read_csv("input/bans_all.csv")       # v1 - irregular grid, updated to sep 2019
years = df.year.values

X = df.drop(['reference', 'datestring', 'year', 'month', 'datenumber'], axis='columns').values
#print(X)

#only samples from year 1991-present
yearsub = years>=1991
X = X[yearsub, :]

############ Build random datasets
# Generate random datasets using Markov chains
nreps = 10
randomDatasets = buildRandomDatasets(X,nreps)


############ Calculate AMI(i,j) for all i,j and datasets



