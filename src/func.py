import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import LeaveOneOut
from multiprocessing import Pool
def train(m,LR,EPOCH,s,x_tr,x_te,y_tr,y_te,f,net):

    loss_func = nn.BCELoss()
    torch.manual_seed(s)
    if net==0:
        net=nn.Sequential(
            nn.Linear(f, m),
            nn.Tanh(),
            nn.Linear(m, 1),
            nn.Sigmoid())
    net.train()
        
    optimizer=torch.optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.99)) 
    for t in range(EPOCH):  
        optimizer.zero_grad()
        loss = loss_func(net(x_tr), y_tr)
        loss.backward()
        optimizer.step()
    net.eval()
    return loss_func(net(x_te),y_te).item(),net
def loadTS(infile):
    df=np.concatenate([pd.read_excel(infile,sheet_name = "train").values[:,1:8],
    pd.read_excel(infile,sheet_name = "rexp").values[:,1:8]])
    x=df[:,:6]
    y=df[:,6]
    return x,y
#### new
def tranforce(x):
    if x=="DMA(50MPa)":
        return 50
    elif x=="DMA(100MPa)":
        return 100
    else:
        return 0
        

def loadMS_class(infile):
    df=pd.read_excel(infile,sheet_name="train")
    # col=['Ti', 'Nb', 'Zr', 'Sn', 'Mo', 'Ta','method']
    col=['Ti', 'Nb', 'Zr', 'Sn', 'Mo', 'Ta']
    x=df[col].values
    y=np.concatenate([df["Ms (K)"].values<300,dfe["phase"]],axis=0)
    #x=df[col].values
    #y=df["Ms (K)"].values<300
    return x,y
def trans_x(x,x_max):
    return x/x_max*0.8
def trans_y(y,y_min,y_max):
    return (y-y_min)/(y_max-y_min)*1.6-0.8
def inv_trans_x(x,x_max):
    return x*x_max/0.8
def inv_trans_y(y,y_min,y_max):
    return (y+0.8)/1.6*(y_max-y_min)+y_min
def preprocess(x,y,outfile,switch=1):
    x_max=np.max(x,axis=0)
    x = trans_x(x,x_max)
    y_max=np.max(y)
    y_min=np.min(y)
	    
    if switch:
	    y = trans_y(y,y_min,y_max)

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=0)
    np.save(outfile,[x_train,x_test,y_train,y_test,x_max,y_min,y_max])
def inv_trans(y,y_min,y_max):
    return (y+0.8)/1.6*(y_max-y_min)+y_min
from itertools import product

import numpy as np



class ConfusionMatrixDisplay:
    """Confusion Matrix visualization.
    It is recommend to use :func:`~sklearn.metrics.plot_confusion_matrix` to
    create a :class:`ConfusionMatrixDisplay`. All parameters are stored as
    attributes.
    Read more in the :ref:`User Guide <visualizations>`.
    Parameters
    ----------
    confusion_matrix : ndarray of shape (n_classes, n_classes)
        Confusion matrix.
    display_labels : ndarray of shape (n_classes,), default=None
        Display labels for plot. If None, display labels are set from 0 to
        `n_classes - 1`.
    Attributes
    ----------
    im_ : matplotlib AxesImage
        Image representing the confusion matrix.
    text_ : ndarray of shape (n_classes, n_classes), dtype=matplotlib Text, \
            or None
        Array of matplotlib axes. `None` if `include_values` is false.
    ax_ : matplotlib Axes
        Axes with confusion matrix.
    figure_ : matplotlib Figure
        Figure containing the confusion matrix.
    """
    def __init__(self, confusion_matrix, *, display_labels=None):
        self.confusion_matrix = confusion_matrix
        self.display_labels = display_labels

    
    def plot(self, *, include_values=True, cmap='viridis',
             xticks_rotation='horizontal', values_format=None, ax=None):
        """Plot visualization.
        Parameters
        ----------
        include_values : bool, default=True
            Includes values in confusion matrix.
        cmap : str or matplotlib Colormap, default='viridis'
            Colormap recognized by matplotlib.
        xticks_rotation : {'vertical', 'horizontal'} or float, \
                         default='horizontal'
            Rotation of xtick labels.
        values_format : str, default=None
            Format specification for values in confusion matrix. If `None`,
            the format specification is 'd' or '.2g' whichever is shorter.
        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.
        Returns
        -------
        display : :class:`~sklearn.metrics.ConfusionMatrixDisplay`
        """
        #check_matplotlib_support("ConfusionMatrixDisplay.plot")
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        cm = self.confusion_matrix
        n_classes = cm.shape[0]
        self.im_ = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        self.text_ = None
        cmap_min, cmap_max = self.im_.cmap(0), self.im_.cmap(256)

        if include_values:
            self.text_ = np.empty_like(cm, dtype=object)

            # print text with appropriate color depending on background
            thresh = (cm.max() + cm.min()) / 2.0

            for i, j in product(range(n_classes), range(n_classes)):
                color = cmap_max if cm[i, j] < thresh else cmap_min

                if values_format is None:
                    text_cm = format(cm[i, j], '.2g')
                    if cm.dtype.kind != 'f':
                        text_d = format(cm[i, j], 'd')
                        if len(text_d) < len(text_cm):
                            text_cm = text_d
                else:
                    text_cm = format(cm[i, j], values_format)

                self.text_[i, j] = ax.text(
                    j, i, text_cm,
                    ha="center", va="center",
                    color=color)

        if self.display_labels is None:
            display_labels = np.arange(n_classes)
        else:
            display_labels = self.display_labels

        fig.colorbar(self.im_, ax=ax)
        
        ax.set(xticks=np.arange(n_classes),
               yticks=np.arange(n_classes),
               xticklabels=display_labels,
               yticklabels=display_labels,
               )

        ax.set_ylim((n_classes - 0.5, -0.5))
        plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)

        self.figure_ = fig
        self.ax_ = ax
        return self