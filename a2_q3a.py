# solution adapted from Dantong Huang
import numpy as np
from sklearn.datasets import load_linnerud

data = load_linnerud()

chins = data.data[:,0]
weight = data.target[:,0]
waist = data.target[:,1]
heartrate = data.target[:,2]

chins = chins < np.median(chins)
chins = chins.astype(float)

p_0 = (20 - np.sum(chins))/20
p_1 = np.sum(chins)/20

def gaussian_naive_prob(inst_attr, all_attr, class_lab):
    class_insts = all_attr[np.where(chins==class_lab)]
    mean = np.mean(class_insts)
    var = np.var(class_insts)
    prob = (1/(2*np.sqrt(var)*np.pi))*np.exp(-(inst_attr-mean)**2/(2*var))
    return prob

probs = np.zeros(20)
for i in np.arange(0,20):
    weight_cond_0 = gaussian_naive_prob(weight[i],weight,0)
    waist_cond_0 = gaussian_naive_prob(waist[i],waist,0)
    heartrate_cond_0 = gaussian_naive_prob(heartrate[i],heartrate,0)

    weight_cond_1 = gaussian_naive_prob(weight[i],weight,1)
    waist_cond_1 = gaussian_naive_prob(waist[i],waist,1)
    heartrate_cond_1 = gaussian_naive_prob(heartrate[i],heartrate,1)
    
    like_0 = weight_cond_0*waist_cond_0*heartrate_cond_0*p_0
    like_1 = weight_cond_1*waist_cond_1*heartrate_cond_1*p_1

    probs[i] = like_0 / (like_1+like_0)


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(data.target, chins)
skprobs = clf.predict_proba(data.target)[:,0]

import matplotlib.pyplot as plt

plt.plot(skprobs, probs, 'o')







