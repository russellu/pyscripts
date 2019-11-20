import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


#data = pd.read_csv('C:/shared/flinks/dataset.csv')

data = pd.read_csv('C:/shared/flinks/challenge_train.csv')

ids = data.LoanFlinksId 
uniques = np.unique(ids)

row_list = []

for i in np.arange(0, uniques.shape[0]): # uniques.shape[0]
    print(uniques[i])
    inds = np.where(uniques[i] == ids)[0]
      
    balance = np.zeros([inds.shape[0],1])
    debs = np.zeros([inds.shape[0],1])
    creds = np.zeros([inds.shape[0],1])
    amounts = np.zeros([inds.shape[0],1])
    days_before_request = np.zeros([inds.shape[0],1])
    isDefault = data.IsDefault[inds[0]]
    loan_amount = data.LoanAmount[inds[0]]
    loan_date = data.LoanDate[inds[0]]
    loan_id = data.LoanFlinksId[inds[0]]
    print(i)
    dates = []
    for j in np.arange(0,inds.shape[0]):
        #print(data.TrxDate[inds[j]])
        #print(data.LoanFlinksId[inds[j]])
        balance[j] = data.Balance[inds[j]]
        debs[j] = data.Debit[inds[j]]
        creds[j] = data.Credit[inds[j]]
        amounts[j] = data.Amount[inds[j]]
        days_before_request[j] = data.DaysBeforeRequest[j]
        dates.append(data.TrxDate[inds[j]])
        
    dates.reverse()
    amounts = np.flipud(amounts)
    days_before_request = np.flipud(days_before_request)
    
    listdata = [['id',loan_id],\
                ['dates',dates],\
                ['transaction_amount',amounts],\
                ['days_before_request',days_before_request],\
                ['loan_amount',loan_amount],\
                ['loan_date',loan_date],\
                ['isDefault',isDefault]]
    
    row_list.append(listdata)
    

#df = pd.DataFrame(row_list)
#df.to_csv('C:/shared/flinks/orig_dataset.csv')

#for i in np.arange(0,10500):
    #print(row_list[i][2])
    #row_list[i][6] = []

#df_test = pd.DataFrame(row_list)
#df_test.to_csv('C:/shared/flinks/dataset.csv')
#df_test.to_pickle('C:/shared/flinks/dataset_labeled10500.pkl')

"""
data = pd.read_pickle('C:/shared/flinks/dataset_unlabeled.pkl')

"""














