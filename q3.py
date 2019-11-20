def q3a():
    f=open("gnb_results.txt","w+")
    data = load_linnerud()
    Chins=data.data[:,0]
    Weight=data.target[:,0]
    Waist=data.target[:,1]
    Heartrate=data.target[:,2]
    med=np.median(Chins)
    dischin=np.where(Chins>med,0,1)
    
    def gnb(a,b,classl):
        aa=a[np.where(dischin == classl)]
        mean=np.mean(aa)
        var=np.var(aa)
        exponent = np.exp(-(b-mean)**2 / (2 * var ))
        return exponent/ np.sqrt(2 * np.pi * var)


       
    p = np.sum(dischin)/len(Chins)
    p0 = 1.0 -p
    for i in range(20):
        gnbweight1=gnb(Weight,Weight[i],1)
        gnbwaist1=gnb(Waist,Waist[i],1)
        gnbHeartrate1=gnb(Heartrate,Heartrate[i],1)

        gnbweight0=gnb(Weight,Weight[i],0)
        gnbwaist0=gnb(Waist,Waist[i],0)
        gnbHeartrate0=gnb(Heartrate,Heartrate[i],0)

        prediction = gnbweight1*gnbwaist1*gnbHeartrate1*p
        predictiondis = gnbweight0*gnbwaist0*gnbHeartrate0*p0

        prob=str(prediction/(prediction+predictiondis))
        print(prob)
        f.write(f"P(chinups=1|instance_{i+1})={prob}"'\n')
    f.close()
    
q3a()