from pomegranate import BayesianNetwork
import numpy as np
import math
import seaborn, time
seaborn.set_style('whitegrid')
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("bans_all.csv")       # v1 - irregular grid, updated to sep 2019
years = df.year.values

X = df.drop(['reference','datestring','year','month','datenumber'], axis='columns').values
#print(X[0:10, :])
#print(X.shape)

#only samples from year 1991-present
yearsub = years>=1991
X = X[yearsub, :]
#print(X.shape)