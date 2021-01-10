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
from models import CNNMnist,CNNCifar
from utils import get_dataset, average_weights, exp_details
import cmfl

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    start_time = time.time()

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
    global_model.to(device)
    global_model.train()
    print(global_model)
    V2=CNNCifar(args=args)
    V2.to(device)
    # copy weights
    global_weights = global_model.state_dict()
    new_global_weights = new_global_model.state_dict()
    chafen_global_weights=chafen_global_model.state_dict()
    # Training
    train_loss, test_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    co=0
    x=0
    E=[]
    
    for epoch in tqdm(range(50)):
        

        x+=1
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')
        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        list_acc, list_loss = [], []
        global_model.eval()
        client=0
        for idx in idxs_users:
            c=0
            client+=1
            local_model = LocalUpdate(args=args, dataset=train_dataset,idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch) 

            local_losses.append(copy.deepcopy(loss))
            acc = local_model.inference1(model=global_model)
            print("第{}轮，全局模型在第{}个客户端的准确率:".format(epoch+1,client),acc)
            list_acc.append(acc)
            if x>1:
                V2.load_state_dict(copy.deepcopy(w))
                w,c=cmfl.main(chafen_global_model,global_model,V2)
                co+=c
        
            local_weights.append(copy.deepcopy(w))
        test_accuracy.append(sum(list_acc)/len(list_acc))
        train_loss.append(sum(local_losses) / len(local_losses))


        global_weights = global_model.state_dict()
        new_global_weights = average_weights(local_weights)
        for k in global_weights.keys():
            chafen_global_weights[k]=new_global_weights[k]-global_weights[k]
        chafen_global_model.load_state_dict(chafen_global_weights)
        global_model.load_state_dict(new_global_weights)
    
    






    print(train_loss)
    print(test_accuracy)
    print(co)
            