'''
Created on Aug 11, 2020

@author: 26sra
'''

import sys #@UnusedImport
import numpy as np
import torch
import os
import time
import functools
import torch.nn as nn
import torch.nn.functional as F #@UnusedImport
import datetime
from itertools import permutations, product #@UnusedImport
from kaggle_environments.envs.halite.helpers import * #@UnusedWildImport
from kaggle_environments import make #@UnusedImport
from random import choice #@UnusedImport

MAX_EPISODES_MEMORY = 512
EPISODE_STEPS = 400
BOARD_SIZE = 21
PLAYERS = 4
GAME_BATCH_SIZE = 8
TRAIN_BATCH_SIZE = 1024
GAME_POOL_SIZE = 16
LEARNING_RATE = 0.1
CHANNELS = 7
MOMENTUM  = 0.9
EPOCHS = 8
WEIGHT_DECAY = 5e-4
TS_FTR_COUNT = 1 + PLAYERS*2 
GAME_COUNT = 10000
TIMESTAMP = "2020_08_11_09.34.59.899412"


class DQN(nn.Module):
    def __init__(self, conv_layers, fc_layers, fc_volume, filters, kernel, stride, pad, ts_ftrs):
        super(DQN, self).__init__()
        
        self._conv_layers = []
        self._relus = []
        self.trained_examples = 0
        height = DQN._compute_output_dim(BOARD_SIZE, kernel, stride, pad)
        for i in range(conv_layers):
            layer = nn.Conv2d(
                CHANNELS if i==0 else filters,   # number of in channels (depth of input)
                filters,    # out channels (depth, or number of filters)
                kernel,     # size of convolving kernel
                stride,     # stride of kernel
                pad)        # padding
            nn.init.xavier_uniform_(layer.weight)
            relu = nn.ReLU()
            
            self._conv_layers.append(layer)
            self._relus.append(relu)
            # necessary to register layer
            setattr(self, "_conv{0}".format(i), layer)
            setattr(self, "_relu{0}".format(i), relu)
            if i!=0:
                height = DQN._compute_output_dim(height, kernel, stride, pad)
            
        
        self._fc_layers = []
        self._sigmoids = []
        for i in range(fc_layers):
            layer = nn.Linear(
                (height * height * filters + ts_ftrs) if i==0 else fc_volume, # number of neurons from previous layer
                fc_volume # number of neurons in output layer
                )
            nn.init.xavier_uniform_(layer.weight)
            sigmoid = nn.Sigmoid()
            self._fc_layers.append(layer)
            self._sigmoids.append(sigmoid)
            
            # necessary to register layer
            setattr(self, "_fc{0}".format(i), layer)
            setattr(self, "_sigmoid{0}".format(i), sigmoid)
            
        self._final_layer = nn.Linear(
                fc_volume,
                1)
        
    def forward(self, geometric_x, ts_x):
        y = self._conv_layers[0](geometric_x)
        y = self._relus[0](y)
        for layer, activation in zip(self._conv_layers[1:], self._relus[1:]):            
            y = layer(y)
            y = activation(y)
        
        y = y.view(-1, y.shape[1] * y.shape[2] * y.shape[3])
        y = torch.cat((y, ts_x), dim=1) #@UndefinedVariable
        for layer, activation in zip(self._fc_layers, self._sigmoids):
            y = layer(y)
            y = activation(y)
        
        return self._final_layer(y)
    
    @staticmethod
    def _compute_output_dim(w, f, s, p):
        for el in [w,f,s,p]:
            DQN._check_and_convert(el)
            if not isinstance(el, int):
                raise ValueError()
        ret = (w-f+2*p)/s + 1
        return DQN._check_and_convert(ret)
    
    @staticmethod
    def _check_and_convert(f):
        if not float(f).is_integer():
            raise ValueError("error on {0}".format(f))
        return int(f)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #@UndefinedVariable
huber = nn.SmoothL1Loss()
dqn = DQN(
    10, # number of conv layers
    2,  # number of fully connected layers at end
    32, # number of neurons in fully connected layers at end
    8,  # number of filters for conv layers (depth)
    3,  # size of kernel
    1,  # stride of the kernel
    0,  # padding
    TS_FTR_COUNT# number of extra time series features
    ).to(device)  
dqn.load_state_dict(torch.load("{0}/dqn_{0}.nn".format(TIMESTAMP)))

optimizer = torch.optim.SGD( #@UndefinedVariable
    dqn.parameters(), 
    lr=LEARNING_RATE, 
    momentum=MOMENTUM, 
    weight_decay=WEIGHT_DECAY)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #@UndefinedVariable

def train(model, criterion, dataset):
    model.train()
    for e in range(EPOCHS):
        idxs = list(range(dataset.episodes_loaded))
        np.random.shuffle(idxs)
        for j, i in enumerate(range(0, len(idxs), TRAIN_BATCH_SIZE)):
            train_idx = idxs[i:i+TRAIN_BATCH_SIZE]
            y_pred = model(
                dataset.g[train_idx], 
                dataset.t[train_idx])
            loss = criterion(y_pred.view(-1), dataset.q[train_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
            print("Epoch: {}, Train batch iteration: {}, Loss: {}".format(e, j, loss.item()))
        torch.save(model.state_dict(), "{0}/dqn_e{1}_{0}.nn".format(TIMESTAMP, e))

class Dataset:
    def __init__(self):
        self.g = torch.zeros(
            (0, CHANNELS, BOARD_SIZE, BOARD_SIZE), 
            dtype=torch.float).to(device) #@UndefinedVariable
                 
        self.t = torch.zeros(
            (0, TS_FTR_COUNT), 
            dtype=torch.float).to(device) #@UndefinedVariable
             
        self.q = torch.zeros(
            0, 
            dtype=torch.float).to(device) #@UndefinedVariable
        
        self.episodes_loaded = 0
        
def main():
    fnames = []
    for gid in range(0, GAME_COUNT, GAME_BATCH_SIZE):
        for pid in range(4):   
            append = 'p{0}g{1}_{2}'.format(pid, gid, TIMESTAMP)
            tup = ('{0}/geo_ftrs_{1}.tensor'.format(TIMESTAMP, append), 
                   '{0}/ts_ftrs_{1}.tensor'.format(TIMESTAMP, append),
                   '{0}/q_values_{1}.tensor'.format(TIMESTAMP, append))
            exists = [os.path.exists(f) for f in tup]
            if all(exists):
                fnames.append(tup)
    np.random.shuffle(fnames)
    
    ds = Dataset()
    for i, tup in enumerate(fnames):
        if ds.episodes_loaded > MAX_EPISODES_MEMORY:
            train(dqn, huber, ds)
            ds = Dataset()
            
        geo = torch.load(tup[0])
        ds.g = torch.cat((ds.g, geo))
            
        ts = torch.load(tup[1])
        ds.t = torch.cat((ds.t, ts))
        
        q = torch.load(tup[2])
        ds.q = torch.cat((ds.q, q))
        
        ds.episodes_loaded += q.shape[0]
        
    if ds.episodes_loaded > 0:
        train(dqn, huber, ds)

main()