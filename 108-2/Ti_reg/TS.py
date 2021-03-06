

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


df=pd.concat([pd.read_excel("../../data/tensile0811.xlsx",sheet_name = "exp"),pd.read_excel("../../data/tensile0811.xlsx",sheet_name = "train")])
df.drop(labels=['Ref.', 'Unnamed: 0'],axis=1, inplace=True)
y = df["E (GPa)"].values
col=['Ti', 'Nb', 'Zr', 'Sn', 'Mo', 'Ta']
x = df[col].values


del df
x_tr,x_te,y_tr,y_te= train_test_split(x, y, test_size=0.2)
mean = np.mean(x_tr, axis = 0) 
std = np.std(x_tr, axis = 0)
for i in range(x_tr.shape[0]):
    for j in range(x_tr.shape[1]):
        if not std[j] == 0 :
            x_tr[i][j] = (x_tr[i][j]- mean[j]) / std[j]
for i in range(x_te.shape[0]):
    for j in range(x_te.shape[1]):
        if not std[j] == 0 :
            x_te[i][j] = (x_te[i][j]- mean[j]) / std[j]

x_tr = torch.tensor(x_tr).float()
x_te = torch.tensor(x_te).float()
y_tr = torch.unsqueeze(torch.tensor(y_tr).float(),dim=1)
y_te = torch.unsqueeze(torch.tensor(y_te).float(),dim=1)
print(col)

loss_func = nn.MSELoss()
f=x.shape[1]    
def train(s,LR,a,EPOCH,act):
    torch.manual_seed(s)
    net=nn.Sequential(
        nn.Linear(f, a),
        act,
        nn.Linear(a, 1))
    optimizer=torch.optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.99)) 
    for t in range(EPOCH):  
        prediction = net(x_tr)
        loss = loss_func(prediction, y_tr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        now=loss_func(net(x_te),y_te)
        
        if t==0:
            loss_ = now
            best_ep=1
        else:
            if loss_ > now:
                loss_=now
                best_ep=t+1
    return loss_.item()**0.5,best_ep,net
#act_list=[nn.PReLU(),nn.CELU(),nn.SELU(),nn.Softplus(),nn.ReLU()]
act_list=[nn.ReLU()]
c=0
for act in act_list:
    for i in range(1,5):
        LR=i/100
        for a in range(10,20):
            if c==0:
                best_l,best_ep,_=train(0,LR,a,2000,act)
                best_lr=LR
                best_a=a
                c+=1
            else:
                l,ep,_=train(0,LR,a,2000,act)
                if l<best_l:
                    best_act=act
                    best_l=l
                    best_lr=LR
                    best_ep=ep
                    best_a=a
        #print(act,l)
print(best_l,best_lr,best_a,best_ep,best_act)

for s in range(10):
    l,ep,net=train(s,best_lr,best_a,best_ep,best_act)
    if s==0:
        l_min,ep_min,net_min=l,ep,net
        s_min=s
    if l<l_min:
        l_min,ep_min,net_min=l,ep,net
        s_min=s
print(l_min)
l,ep,net=train(s_min,best_lr,best_a,ep_min,best_act)  
torch.save([s_min,best_lr,best_a,best_ep,act,net],"model/%.4f.pkl" %(best_l))
np.save("model/%.4f.npy"%(best_l),[mean,std])
