from pomegranate import BayesianNetwork
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
seaborn.set_style('whitegrid')


df = pd.read_csv("bans_all.csv")       # v1 - irregular grid, updated to sep 2019
years = df.year.values

X = df.drop(['reference','datestring','year','month','datenumber'], axis='columns').values
#print(X[0:10, :])
#print(X.shape)

#only samples from year 1991-present
yearsub = years>=1991
X = X[yearsub, :]
#print(X.shape)

smoother = 1.0      # final version uses 1.0
model = BayesianNetwork.from_samples(X, algorithm='exact-dp', pseudocount=smoother)
print('model summary', model.copy())

'''
# Save model so we can load at a later time
with open("bansmodel.json", "w") as outfile: outfile.write(model.to_json())
#read back model
#BayesianNetwork.from_json("bansmodel.json")
'''

print(model.structure)
plt.figure(figsize=(7, 7))
model.plot()
plt.show()

# Example inference: Predict based on observed nodes
# 0 - clear; 1 - ban start; 2 - ban continuation
calbayog = None     # X_0
cambatutay = 2      # X_1
irong = 2           # X_2
maqueda = None      # X_3
villareal = None    # X_4
daram = None        # X_5
biliran = None      # X_6
carigara = 2        # X_7
coastalleyte = None # X_8
calubian = None     # X_9
sanpedro = None     # X_10

sitenames = ['calbayog', 'cambatutay', 'irong', 'maqueda', 'villareal', 'daram', 'biliran', 'carigara', 'coastalleyte',
                      'calubian', 'sanpedro']
modpred = model.predict([[calbayog, cambatutay, irong, maqueda, villareal, daram, biliran, carigara, coastalleyte,
                      calubian, sanpedro]])
#print(modpred)
print('Prediction:')
for i in range(0, len(sitenames)):
    print(sitenames[i], ': ', modpred[0][i])
