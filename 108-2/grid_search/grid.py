#!/usr/bin/env python
# coding: utf-8

# In[1]:




import torch
from torch.autograd import Variable
from torch.nn import functional as F
import pandas as pd
import numpy as np
import os
from collections import Counter
from multiprocessing import Pool
from  func import BN,load,output


# In[2]:




mean_T,std_T,model_T,_=load("../Ti_reg/model")
#output(np.array([[65,11.6,11.4,12,0,0]]),mean_T,std_T,model_T)
mean_M,std_M,model_M,_=load("../Ms_reg/model")
#output(np.array([[62.5,26,4,7.5,0,0,0]]),mean_M,std_M,model_M)

point=[]
for Ti in range(41,101):
    for Nb in range(0,20):
        if Nb+Ti>100:
            break
        for Zr in range(0,20):
            if Nb+Ti+Zr>100:
                break
            for Sn in range(0,20):
                if Nb+Ti+Zr+Sn>100:
                    break
                for Mo in range(0,20):
                    if Nb+Ti+Zr+Sn+Mo>100 or Nb+Ti+Zr+Sn+Mo<=80:
                        break
                    else:
                        point.append([Ti,Nb,Zr,Sn,Mo,100-(Nb+Ti+Zr+Sn+Mo),0])
point=np.array(point)



# In[3]:



#point[0,:-1]
#TS=result(point[:,:-1],mean_T,std_T,model_T)
T=[];M=[]
if __name__ == '__main__':
    p = Pool(processes=5) 
    for i in range(len(mean_M)):
        t=p.apply_async(output,(point[:,:-1],mean_T[i],std_T[i],model_T[i],))
        m=p.apply_async(output,(point[:,:],mean_M[i],std_M[i],model_M[i],))
        T.append(t)
        M.append(m)
    p.close()
    p.join()


# In[4]:


tensile=[]
Ms=[]
for i in range(len(T)):
    tensile.append(T[i].get())
    Ms.append(M[i].get())
tensile=np.array(tensile)
Ms=np.array(Ms)
Ts_mean=np.mean(tensile,axis=0)
Ms_mean=np.mean(Ms,axis=0)
val_ind=[]
for i,ele in enumerate(Ms_mean<300):
    if ele:
        val_ind.append(i)
Ts_mean=Ts_mean[val_ind]
Ms_mean=Ms_mean[val_ind]
point=point[val_ind]


# In[5]:


mat=np.concatenate([point,Ts_mean.reshape(-1,1),Ms_mean.reshape(-1,1)],axis=1)


# In[6]:


#Pearson Correaltion Coefficient
cov=np.cov(mat.T)


# In[7]:


coef=np.empty([6,2])
for i in range(6):
    coef[i,0]=cov[i,7]/np.sqrt(cov[i,i]*cov[7,7])
    coef[i,1]=cov[i,8]/np.sqrt(cov[i,i]*cov[8,8])
    
    


# In[8]:


pd.DataFrame(coef,columns=["tensile","Ms"]).to_excel("correlation.xlsx")


# In[36]:





# In[12]:


df=pd.read_excel("../../test/0122.xlsx").values
df=np.concatenate([df,np.zeros([len(df),1])],axis=1)
def simple_output(data,mean,std,model):
    ret=np.empty([len(mean),len(data)])

    for i in range(len(mean)):
        #print(mean[i],std[i],model[i])
        ret[i]=output(data,mean[i],std[i],model[i])
    return ret
        
samp_T=simple_output(df[:,:-1],mean_T,std_T,model_T)
samp_M=simple_output(df,mean_M,std_M,model_M)


# In[13]:


df=np.concatenate([df[:,:-1],np.mean(samp_T,axis=0).reshape(-1,1),np.std(samp_T,axis=0).reshape(-1,1),
                  np.mean(samp_M,axis=0).reshape(-1,1),np.std(samp_M,axis=0).reshape(-1,1)],axis=1)


# In[14]:


with pd.ExcelWriter('../../test/testing_output.xlsx') as writer:  
    pd.DataFrame(df).to_excel(writer,sheet_name="original")
    pd.DataFrame(samp_T.T).to_excel(writer,sheet_name="tensile")
    pd.DataFrame(samp_M.T).to_excel(writer,sheet_name="Ms")


# In[74]:


df=pd.read_excel("../../data/Ms0811.xlsx")
y=df["Ms (K)"].values
df["method"]=df["method"].map({'DSC': 0,"DMA(50MPa)":50,"DMA(100MPa)":100,"DMA(0MPa)":0})
df["method"].fillna(0,inplace=True)
df=df.iloc[:,1:8].values


# In[75]:


for i in range(len(mean_M)):
    print(i,np.sqrt(np.sum((output(df,mean_M[i],std_M[i],model_M[i])-y)**2)/len(y)))
    print(name[2*i])


# In[ ]:





# In[ ]:





# In[ ]:




