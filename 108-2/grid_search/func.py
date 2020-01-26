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

def create3(M):
    lis=[]
    for i in range(M+1):
        j=i
        while j<=M:
            lis.append([i,(j-i),int(M-j)])
            j+=1
    return lis
def create2(M):
    lis=[]
    for i in range(M+1):
        lis.append([i,M-i])
    return lis
def classifyMs(x):
    path=os.getcwd()+"/Ms"
    name_list=[]
    for name in os.listdir(path):
        if ".pkl" in name:
            name_list.append(name)
    pre=[]
    correct=[0 for i in range(len(name_list))]
    for name in name_list:
        name=os.getcwd()+"/Ms/"+name
        net2=torch.load(name).cpu()
        outputs = net2(torch.from_numpy(x).float())
        _, prediction = torch.max(F.softmax(outputs,dim=1), 1)
        pre_num=prediction.numpy()
        pre.append(pre_num)
    arr=np.array(pre)
    pre.clear()
    result=[Counter(arr[:,i]).most_common(1)[0][0] for i in range(len(x))]
    return torch.Tensor(result)
def BN(data,mean,std):
    data_o=data.copy().astype("float")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            #if not std[j] == 0 :
            data_o[i][j] = (data[i][j]- mean[j]) / std[j]
    return data_o

def load(path):
    mean=[]
    std=[]
    model=[]
    name=[]
    for ele in os.listdir(path):
        abs_path=path+"/"+ele
        if "npy" in ele:
            m,s=np.load(abs_path)
            mean.append(m)
            std.append(s)
        elif "pkl" in ele:
            model.append(torch.load(abs_path)[-1])
        name.append(abs_path)
    return mean,std,model,name

def output(comp,mean,std,model):
    return model(torch.tensor(BN(comp,mean,std),dtype=torch.float32)).view(-1).cpu().detach().numpy()
    #return model(torch.tensor(BN(comp,mean,std),dtype=torch.float32)).view(-1).cpu().detach().numpy()
    
    
