from sklearn.datasets import load_diabetes
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_digits
from sklearn.datasets import load_wine
from sklearn.datasets import load_linnerud
import matplotlib.pyplot as plt
import numpy as np  

dia = load_diabetes() 
cancer = load_breast_cancer() 
wine = load_wine()
lin = load_linnerud()

# age, sex, BMI, BP

phys = lin.target
exc = lin.data

weight = phys[:,0]
waist = phys[:,1]
hr = phys[:,2]

chins = exc[:,0]
situps = exc[:,1]
jumps = exc[:,2]

nom_weight = []
nom_waist = []
nom_hr = []
nom_chins = [] 
nom_situps = []
nom_jumps = []
for i in np.arange(0,weight.shape[0]):
    if weight[i] > 180: 
        nom_weight.append('heavy')
    else:
        nom_weight.append('light')
    
    if waist[i] > 35:
        nom_waist.append('wide')
    else:
        nom_waist.append('thin')
        
    if hr[i] > 55:
        nom_hr.append('fast')
    else:
        nom_hr.append('slow')
        
    if chins[i] > 10:
        nom_chins.append('yes')
    else:
        nom_chins.append('no')      
    if situps[i] > 150:
        nom_situps.append('yes')
    else:
        nom_situps.append('no')
    if jumps[i] > 50:
        nom_jumps.append('yes')
    else:
        nom_jumps.append('no')
    
for i in np.arange(0,20):
    print(weight[i])










