from pomegranate import *
import numpy as np


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
    return Xrands.astype(int)
