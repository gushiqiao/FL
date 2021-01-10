#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
from options import args_parser
from update import LocalUpdate, test_inference
from models import CNNCifar
from utils import get_dataset, average_weights, exp_details
import chafen
import math
if __name__ == '__main__':
    start_time = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    device = 'cuda' 

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    global_model = CNNCifar(args=args)

    new_global_model = CNNCifar(args=args)
    new_global_model.to(device)
    new_global_model.train()
    chafen_global_model = CNNCifar(args=args)
    chafen_global_model.to(device)
    chafen_global_model.train()
    chafen_local_model = CNNCifar(args=args)
    chafen_local_model.to(device)
    chafen_local_model.train()
    global_model.to(device)
    global_model.train()
    print(global_model)
    V2=CNNCifar(args=args)
    V2.to(device)
    # copy weights
    global_weights = global_model.state_dict()
    new_global_weights = new_global_model.state_dict()
    chafen_global_weights=chafen_global_model.state_dict()
    chafen_local_weights=chafen_local_model.state_dict()
    # Training
    train_loss, test_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0
    co1=0
    co2=0
    x=0
    E1=[]
    E2=[]
    l=[]
    my_sum=0
    Acc1=[]
    Acc2=[]
    global_models=[]
    global_models.append(global_model.state_dict())
    r=0.5
    for epoch in tqdm(range(100)):    
        x+=1 
        list_acc, list_loss = [], []
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')
        global_model.train()
        global_weights = global_model.state_dict()   
        m = max(int(args.frac *100), 1)
        idxs_users = np.random.choice(range(100), m, replace=False)
        i=-1

        for idx in idxs_users:
            flag=0
            i+=1
            local_model = LocalUpdate(args=args, dataset=train_dataset,idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
            acc= local_model.inference1(model=global_model)

            local_losses.append(copy.deepcopy(loss))
            list_acc.append(acc)
            if x==1:
                for k in w.keys():
                    chafen_local_weights[k]=w[k]-global_weights[k]
                l.append(chafen_local_weights)
            if x>1:
                V2.load_state_dict(copy.deepcopy(w))
                w,l[i],c1,c2,my_sum1,acc1,acc2,flag=chafen.main(r,flag,x,global_models,chafen_global_model,global_model,V2,local_model.trainloader,l[i],local_model.testloader)
                co1+=c1
                co2+=c2
                my_sum+=my_sum1
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            list_acc.append(acc)

        test_accuracy.append(sum(list_acc)/len(list_acc))
        train_loss.append(sum(local_losses) / len(local_losses))
        global_weights = global_model.state_dict()
        new_global_weights = average_weights(local_weights)
        for k in global_weights.keys():
            chafen_global_weights[k]=new_global_weights[k]-global_weights[k]
        chafen_global_model.load_state_dict(chafen_global_weights)
        global_model.load_state_dict(new_global_weights)
        global_models.append(new_global_weights)



    

    print(train_loss)
    print(test_accuracy)
    print(co1)
    print(co2)
    print(my_sum)
