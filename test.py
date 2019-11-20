import numpy as np
import matplotlib.pyplot as plt
#from IPython import get_ipython
#from sympy import *

A=np.random.randint(2, size=(50, 1000));

B=np.zeros((1000,1000),dtype=float);


for i in range(len(A[0])):
    for j in range(len(A[0])):
        a1=np.sum((A[:,i]-np.mean(A[:,i]))*(A[:,j]-np.mean(A[:,j])));
        a2=np.sqrt(np.sum((A[:,i]-np.mean(A[:,i]))*(A[:,i]-np.mean(A[:,i]))));
        a3=np.sqrt(np.sum((A[:,j]-np.mean(A[:,j]))*(A[:,j]-np.mean(A[:,j]))));
        B[i,j]=a1/(a2*a3);
        
k1=1;
k2=100;
v=0;
rr=[];
while(k2>0):
    for j in range(k2-1,100):
        rr.append(B[k1+k2-100+v-1,j]);
        v=v+1;
    k1=k1+1;
    k2=k2-1;
    v=0;
    
a=[];
v=len(rr);
for n in range(100,0,-1):
    a.append(np.sum(np.abs(rr[(v-n):v])));
    v=v-n;
b=[];
for n in range(99,0,-1):
    b.append(a[n]);

a1=b+a;

#get_ipython().magic(u'matplotlib qt')
plt.figure(1)
#get_ipython().run_line_magic('matplotlib', 'qt')
plt.stem(a1,markerfmt = 'ro', linefmt = 'g--', basefmt = 'm:')
plt.margins(0.1, 0.1)
plt.title('stem of person coefficients')
plt.ylabel('value')
plt.show()


k1=1;
k2=50;
v=0;
rr=[];
while(k2>0):
    for j in range(k2-1,50):
        rr.append(B[k1+k2-50+v-1,j]);
        v=v+1;
    k1=k1+1;
    k2=k2-1;
    v=0;
    
a=[];
v=len(rr);
for n in range(50,0,-1):
    a.append(np.sum(np.abs(rr[(v-n):v])));
    v=v-n;
    
b=[];
for n in range(49,0,-1):
    b.append(a[n]);

a1=b+a;

#get_ipython().magic(u'matplotlib qt')
plt.figure(2)
plt.stem(a1,markerfmt = 'ro', linefmt = 'g--', basefmt = 'm:')
plt.margins(0.1, 0.1)
plt.title('P(x>0.75)=0.73')
plt.ylabel('value')
plt.show()


k1=1;
k2=10;
v=0;
rr=[];
while(k2>0):
    for j in range(k2-1,10):
        rr.append(B[k1+k2-10+v-1,j]);
        v=v+1;
    k1=k1+1;
    k2=k2-1;
    v=0;
    
a=[];
v=len(rr);
for n in range(10,0,-1):
    a.append(np.sum(np.abs(rr[(v-n):v])));
    v=v-n;

b=[];
for n in range(9,0,-1):
    b.append(a[n]);

a1=b+a;

#get_ipython().magic(u'matplotlib qt')
plt.figure(3)
plt.stem(a1,markerfmt = 'ro', linefmt = 'g--', basefmt = 'm:')
plt.margins(0.1, 0.1)
plt.title('P(x>0.75)=0.57')
plt.ylabel('value')
plt.show()






