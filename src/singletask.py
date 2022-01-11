import torch
import pandas as pd
import numpy as np
import random
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from func import loadMS_class,loadTS
import multiprocessing as mp
import time

def splitTs():    
    df = pd.read_excel("../data/tensile.xlsx")
    y = df["E (GPa)"].values
    col=['Ti', 'Nb', 'Zr', 'Sn', 'Mo', 'Ta']
    x = df[col].values
    del df
    x_tr,x_te,y_tr,y_te= train_test_split(x, y, test_size=0.2,random_state=0)
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
    return x_tr,x_te,y_tr,y_te
def splitMs():
    df = pd.read_excel("../data/stability.xlsx")
    col=['Ti', 'Nb', 'Zr', 'Sn', 'Mo', 'Ta']
    x = df[col].values
    y = df["stability"].values
    x_tr,x_te,y_tr,y_te= train_test_split(x, y, test_size=0.2, random_state=0)
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
    return x_tr,x_te,y_tr,y_te
def accuracy(pred,label):
    return np.mean((pred.detach().numpy()>0.5) == label.detach().numpy())
TsDataset = splitTs()
MsDataset = splitMs()
act_list=[nn.PReLU(),nn.CELU(),nn.SELU(),nn.Softplus(),nn.ReLU()]
f = TsDataset[0].shape[1]
lossMSE = nn.MSELoss()
lossBCE = nn.BCELoss()
sig = nn.Sigmoid()
class Train:
    def __init__(self,index,reg):
        self.m = np.random.randint(low=5, high=30)
        self.lr = np.random.uniform(low=0.001, high=0.1)
        self.act = np.random.choice(act_list)
        self.net = nn.Sequential(nn.Linear(f, self.m),self.act,nn.Linear(self.m, 1))
        self.opt = torch.optim.Adam(self.net.parameters(),lr=self.lr)
        self.score = 0
        self.index = index
        self.reg = reg
def train(obj,TsDataset,MsDataset,epoch=2000):
    save = False
    for i in range(epoch):
        if obj.reg:
            x_tr,x_te,y_tr,y_te = TsDataset
            loss = lossMSE(obj.net(x_tr),y_tr)
            mse_te = lossMSE(obj.net(x_te.detach()),y_te.detach()).item()
            obj.score = -mse_te
        else:            
            x_tr,x_te,y_tr,y_te = MsDataset
            loss = lossBCE(sig(obj.net(x_tr)),y_tr)
            acc_tr = accuracy(sig(obj.net(x_tr.detach())).detach().numpy(),y_tr.detach().numpy()).item()
            acc_te = accuracy(sig(obj.net(x_te.detach())).detach().numpy(),y_te.detach().numpy()).item()
            obj.score = acc_te
        # print(obj.score)
        if i == 0:
            if obj.reg:
                best_mse = mse_te
            else:
                best_acc = acc_te
        else:
            if obj.reg:
                if mse_te < best_mse:
                    best_mse = mse_te
                    if best_mse < 100:
                        save = True
            else:
                if acc_te > best_acc:
                    best_acc = acc_te
                    if best_acc > 0.85:
                        save = True
            if save:
                save = False
                if obj.reg:
                    torch.save(obj.net,"../single/reg/%.2f_%s_%d_%.3f_%d.pkl"%(-obj.score,obj.act,obj.m,obj.lr,i+1))
                else:
                    torch.save(obj.net,"../single/cls/%.2f_%s_%d_%.3f_%d.pkl"%(obj.score,obj.act,obj.m,obj.lr,i+1))                        
        obj.opt.zero_grad()
        loss.backward()
        obj.opt.step()        
if __name__ == '__main__':
    num = 10
    regs = [True,False]
    start = time.time()
    for reg in regs:                                       
        trains = [Train(i,reg) for i in range(num)]
        for i in range(100):
            ps = [mp.Process(target=train, args=(trains[k],TsDataset,MsDataset)) for k in range(num)]
            for p in ps:
                p.start()
            for p in ps:
                p.join()
        print(time.time()-start)
            
