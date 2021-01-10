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
import sample

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    start_time = time.time()

    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    device = 'cuda' 

    train_dataset, test_dataset, user_groups = get_dataset(args)

    global_model = CNNCifar(args=args)
    V2=CNNCifar(args=args)
    V2.to(device)

    global_model.to(device)
    global_model.train()
    print(global_model)

    global_weights = global_model.state_dict()

    train_loss, test_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0
    x=0
    my_sum=0
    for epoch in tqdm(range(100)):
        x+=1
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        list_acc, list_loss = [], []
        global_model.eval()
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
            acc=local_model.inference1(model=global_model)

            if x>1:
                V2.load_state_dict(copy.deepcopy(w))
                w,my_sum1=sample.main(global_model,V2,local_model.testloader)
                my_sum+=my_sum1
                
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            list_acc.append(acc)

        
        
        test_accuracy.append(sum(list_acc)/len(list_acc))
        train_loss.append(sum(local_losses) / len(local_losses))

        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)





    print(train_loss)
    print(test_accuracy)   
    print(my_sum)
    print("sample")
  
