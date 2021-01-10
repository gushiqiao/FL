#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch.nn.functional as F
from options import args_parser


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import math

class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84) 
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
    def f(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        return x
class CNNCifar1(nn.Module):
    def __init__(self, args):
        super(CNNCifar1, self).__init__()
        self.conv11 = nn.Conv2d(3, 6, 5)
        self.conv12 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv21 = nn.Conv2d(6, 16, 5)
        self.conv22 = nn.Conv2d(6, 16, 5)
        self.fc11 = nn.Linear(16 * 5 * 5, 120)
        self.fc12 = nn.Linear(16 * 5 * 5, 120)
        self.fc21 = nn.Linear(120, 84)
        self.fc22 = nn.Linear(120, 84)
        self.fc31 = nn.Linear(84, args.num_classes)
        self.fc32 = nn.Linear(84, args.num_classes)
    def forward(self, x):
        x = self.pool(F.relu(self.conv11(x)+self.conv12(x)))
        x = self.pool(F.relu(self.conv21(x)+self.conv22(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc11(x)+self.fc12(x))
        x = F.relu(self.fc21(x)+self.fc22(x))
        x = self.fc31(x)+self.fc32(x)                                                
        return F.log_softmax(x, dim=1)

    def f(self,x):
        x = self.pool(F.relu(self.conv11(x)+self.conv12(x)))
        x = self.pool(F.relu(self.conv21(x)+self.conv22(x)))
        x = x.view(-1, 16 * 5 * 5)
        return x
    def f1(self,x):
        x = self.pool(F.relu(self.conv11(x)))
        x = self.pool(F.relu(self.conv21(x)))
        x = x.view(-1, 16 * 5 * 5)
        return x


class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)  
        self.fc1   = nn.Linear(16*5*5, 120)  
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
 
    def forward(self, x): 
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2)) 
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) 
        x = x.view(x.size()[0], -1)   #展开成一维的
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)        
        return F.log_softmax(x, dim=1)
    def f(self, x): 
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2)) 
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) 
        x = x.view(x.size()[0], -1)
        return F.log_softmax(x, dim=1)
class CNNMnist1(nn.Module):
    def __init__(self, args):
        super(CNNMnist1, self).__init__()
        self.conv11 = nn.Conv2d(1, 6, 5,padding=2)
        self.conv12 = nn.Conv2d(1, 6, 5,padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv21 = nn.Conv2d(6, 16, 5)
        self.conv22 = nn.Conv2d(6, 16, 5)
        self.fc11 = nn.Linear(16 * 5 * 5, 120)
        self.fc12 = nn.Linear(16 * 5 * 5, 120)
        self.fc21 = nn.Linear(120, 84)
        self.fc22 = nn.Linear(120, 84)
        self.fc31 = nn.Linear(84, args.num_classes)
        self.fc32 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv11(x)+self.conv12(x)))
        x = self.pool(F.relu(self.conv21(x)+self.conv22(x)))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc11(x)+self.fc12(x))
        x = F.relu(self.fc21(x)+self.fc22(x))
        x = self.fc31(x)+self.fc32(x)
        return F.log_softmax(x, dim=1)

    def f(self, x):
        x = self.pool(F.relu(self.conv11(x)+self.conv12(x)))
        x = self.pool(F.relu(self.conv21(x)+self.conv22(x)))
        x = x.view(x.size()[0], -1)
        return x
