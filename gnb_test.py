from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_linnerud
import numpy as np

data = load_linnerud()

chins = data.data[:,0]
ychins = np.zeros(chins.shape[0])

ychins[chins>np.median(chins)] = 0
ychins[chins<=np.median(chins)] = 1

gnb = GaussianNB()
gnb.fit(data.target,ychins)

probs = gnb.predict_proba(data.target)