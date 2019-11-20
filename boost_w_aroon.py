import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

# data available at: https://drive.google.com/open?id=1TxyJlii6O0YCmgxOWuMKnpnaqwbBx4R-

def getmvg(data,period):
    mvg = np.zeros([data.shape[0],1])
    
    for i in np.arange(period+1,data.shape[0]):
        if i==period+1:
            mvg[i]=np.mean(data[i-period+1:i])
        else :
            mvg[i] = mvg[i-1] + data[i]/period - data[i-period]/period
        
    return mvg

def getbbands(data,period,stdfactor):
    mvg = getmvg(data,period)
    bbands = np.zeros([2,data.shape[0]])
    for i in np.arange(period+1,data.shape[0]):
        std_i = np.std(data[i-period:i])
        bbands[0,i] = mvg[i] + std_i*stdfactor
        bbands[1,i] = mvg[i] - std_i*stdfactor
    
    return bbands

def aroonosc(data, period):
    aroonvals = np.zeros([2,data.shape[0]])
    for i in np.arange(period+1,data.shape[0]):
        maxind = np.where(data[i-period:i]==np.max(data[i-period:i]))
        minind = np.where(data[i-period:i]==np.min(data[i-period:i]))
       
        aroonvals[0,i] = (maxind[0][0]/period)*100
        aroonvals[1,i] = (minind[0][0]/period)*100
        
    aroonosc = aroonvals[0,:] - aroonvals[1,:]
    return aroonosc
    
currs = ['AUDCAD','AUDCHF','AUDJPY','AUDNZD','AUDSGD','AUDUSD','CADCHF','CADJPY',\
         'CHFJPY','CHFSGD','EURAUD','EURCAD','EURCHF','EURGBP','EURJPY','EURNZD',\
         'EURSGD','EURUSD','GBPAUD','GBPCAD','GBPCHF','GBPJPY','GBPNZD','GBPUSD',\
         'HKDJPY','NZDCAD','NZDCHF','NZDJPY','NZDUSD','SGDJPY','USDCAD','USDCHF',\
         'USDHKD','USDJPY','USDSGD','ZARJPY']

scales = [.0001,.0001,.01,.0001,.0001,.0001,.0001,.01,\
          .01,.0001,.0001,.0001,.0001,.0001,.01,.0001,\
          .0001,.0001,.0001,.0001,.0001,.01,.0001,.0001,\
          .01,.0001,.0001,.01,.0001,.01,.0001,.0001,\
          .0001,.01,.0001,.01]

currcount=0
bdiffs = np.zeros([36,29,40])

for curr in currs:
    csv = pd.read_csv('C:/shared/dukascopy/all30mins/'+curr+'_Hourly_Ask_2009.11.09_2019.11.08.csv')
    bcount=0
    for b in np.arange(5,150,5):
        rawdata = np.zeros([csv.Close.shape[0],4])
        rawdata[:,0] = csv.Open; 
        rawdata[:,1] = csv.High; 
        rawdata[:,2] = csv.Low;
        rawdata[:,3] = csv.Close; 
        
        vols = np.zeros(rawdata.shape[0])
        for i in np.arange(0,vols.shape[0]):
            vols[i]=csv['Volume '][i]    
        
        aroon = aroonosc(rawdata[:,3],b)
        bbands = getbbands(rawdata[:,3],b,3.5)
        bband_diff = rawdata[:,3] - bbands 
        x_data = np.zeros([vols.shape[0],8])
        
        x_data[:,0:4] = rawdata
        x_data[:,4] = vols
        x_data[:,7] = aroon
        x_data[:,5] = bband_diff[0,:]
        x_data[:,6] = bband_diff[1,:] 
        x_data = x_data[150:,:]
        raw_x = rawdata[150:,3]
        y_data = np.zeros([raw_x.shape[0]])
        for i in np.arange(0,raw_x.shape[0]-8):
            diffs = raw_x[i+1:i+8] - raw_x[i+1]
            y_data[i] = np.max(diffs)-np.min(diffs)
        
        n_trains = int(y_data.shape[0]*0.9)
        x_train = x_data[0:n_trains,:]
        y_train = y_data[0:n_trains]
        x_test = x_data[n_trains:,:]
        y_test = y_data[n_trains:]
             
        from sklearn.ensemble import GradientBoostingRegressor
        
        est = GradientBoostingRegressor(loss='huber',n_estimators=600,max_depth=1)
        
        est.fit(x_train,y_train)
        preds = est.predict(x_test)
        
        #plt.subplot(1,2,1)
        #plt.plot(preds,y_test,'.')
        #plt.title(np.corrcoef(preds.T,y_test.T)[0,1])
        
        inds = np.argsort(preds)
        ytest_sort = y_test[inds]
        
        # validate - get top epochs
        shift_inds = inds
        shift_inds = np.delete(shift_inds,np.where(shift_inds>(raw_x.shape[0]-n_trains)-100))
        
        newraw_x = raw_x[n_trains:]
        sorted_epochs = np.zeros([shift_inds.shape[0],12])
        for i in np.arange(0,shift_inds.shape[0]):
            sorted_epochs[i,:] = newraw_x[shift_inds[i]:(shift_inds[i]+12)]
            
        sorted_epochs = np.cumsum(np.diff(sorted_epochs,axis=1),axis=1)
        arcount=0
        for ar in np.arange(10,1010,25):
            bdiffs[currcount,bcount,arcount] = (np.mean(sorted_epochs[-ar:,8],axis=0) \
                            - np.mean(sorted_epochs[0:ar,8],axis=0))/scales[currcount]
            arcount=arcount+1
        bcount = bcount + 1
        print(bcount)
    currcount = currcount+1
    print(curr + ' ' +str(np.mean(bdiffs[currcount-1,:,1:8])))
    
# with, n_est=100, lrate=0.02, max_depth=1, == 3.41
# with, n_est=100, lrate=0.1, max_depth=1, == 6.96
# with, n_est=600, lrate=0.1, max_depth=1, == 7.1  
    
#plt.subplot(1,2,2)
#plt.plot(np.mean(sorted_epochs[0:1000,:],axis=0))
#plt.plot(np.mean(sorted_epochs[-1000:,:],axis=0))

# a more realistic simulation - buy and sell using the classifier at every step
"""
goods = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,19,20,21,23,24,25,26,27,\
         28,29,30,31,32,33,34,35,36,37,38,39,40,41,42]

score = np.mean(bdiffs[:,:,1:8])
print(score)


for i in np.arange(0,36):
    plt.subplot(6,10,i+1)
    plt.imshow(bdiffs[i,:,:],vmin=-20,vmax=20)
    plt.title(currs[i])
    plt.xticks(ticks=[]); 
"""
