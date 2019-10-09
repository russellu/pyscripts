from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np 
import matplotlib.pyplot as plt 

hours = [0.5,0.75,1,1.25,1.5,1.75,1.75,2,2.25, \
         2.5,2.75,3,3.25,3.5,4,4.25,4.5,4.75,5,5.5]; 
passfail = [0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1];
#passfail = [1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1];

hours = np.expand_dims(np.asarray(hours),1)
passfail = np.asarray(passfail)

clf = LogisticRegression(random_state=0, solver='newton-cg',
                       multi_class='ovr').fit(hours, passfail)

new_instance = np.zeros([1,1]); new_instance[0,0] = 5
prob_5hrs = clf.predict_proba(new_instance)
new_instance = np.zeros([1,1]); new_instance[0,0] = 2
prob_2hrs = clf.predict_proba(new_instance)

x_axis = np.arange(0,7,0.5)
sigmoid = 1/(1+np.exp(-clf.intercept_-x_axis*clf.coef_))

plt.plot(x_axis,sigmoid.T)
plt.plot(hours,passfail,'o')
#plt.plot(5,prob_5hrs[0,1],'o'); plt.plot(2,prob_2hrs[0,1],'o')
plt.plot([0,6],[0.5,0.5])
















