from pomegranate import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
seaborn.set_style('whitegrid')
import os


if not os.path.exists('output'):
    os.makedirs('output')


############ Build raining data
df = pd.read_csv("input/bans_all.csv")       # v1 - irregular grid, updated to sep 2019
years = df.year.values

X = df.drop(['reference','datestring','year','month','datenumber',
             ], axis='columns').values
print(X)

# only samples from year 1991-present
yearsub = years>=1991
X = X[yearsub, :]
print(X)


############ Learn parameters for the null model (all nodes independent)
smoother = 1      # final version uses 1.0
nullstruct = ((), (), (), (), (), (), (), (), (), (), ())
nullmodel = BayesianNetwork.from_structure(X, structure=nullstruct, pseudocount=smoother)
print('model summary', nullmodel.copy())
print(nullmodel.structure)
fig = plt.figure(figsize=(7, 7))
nullmodel.plot()
plt.show()


############ Performance on training and validation data
validdf = pd.read_csv("input/bans_validation.csv")       # v1 - irregular grid, updated to sep 2019
X2 = validdf.drop(['reference','datestring','year','month'], axis='columns').values
print(X2)

# LogP(data|model)
print('log P(training data|null model')
print(sum(nullmodel.log_probability(X)))
print('log P(validation data|null model')
print(sum(nullmodel.log_probability(X2)))