from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils import get_dataset
from options import args_parser
from update import test_inference
from torch.autograd import Variable
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from functools import reduce
from utils import get_dataset
from options import args_parser
from update import test_inference
from models import CNNMnist,CNNMnist1,CNNCifar,CNNCifar1
import operator
from thop import profile

def main(chafen_model,V1,V2) :
    args = args_parser()
    device="cuda"
    c=0
    r=0.5
    chafen=CNNCifar(args=args)
    V1.to(device)
    V2.to(device)
    chafen.to(device)

    V1_weight=V1.state_dict()
    V2_weight=V2.state_dict()
    chafen_model_weight=chafen_model.state_dict()
    chafen_weight=chafen.state_dict()

    chafen_weight['conv1.weight']=V2_weight['conv1.weight']-V1_weight['conv1.weight']
    chafen_weight['conv1.bias']=V2_weight['conv1.bias']-V1_weight['conv1.bias']
    chafen_weight['conv2.weight']=V2_weight['conv2.weight']-V1_weight['conv2.weight']
    chafen_weight['conv2.bias']=V2_weight['conv2.bias']-V1_weight['conv2.bias']
   
    chafen_weight['fc1.weight']=V2_weight['fc1.weight']-V1_weight['fc1.weight']
    chafen_weight['fc1.bias']=V2_weight['fc1.bias']-V1_weight['fc1.bias']
    chafen_weight['fc2.weight']=V2_weight['fc2.weight']-V1_weight['fc2.weight']
    chafen_weight['fc2.bias']=V2_weight['fc2.bias']-V1_weight['fc2.bias']
    chafen_weight['fc3.weight']=V2_weight['fc3.weight']-V1_weight['fc3.weight']
    chafen_weight['fc3.bias']=V2_weight['fc3.bias']-V1_weight['fc3.bias']

    chafen.load_state_dict(chafen_weight)
    
    sign_sum = 0
    sign_size = 0
    
    for k in chafen_weight.keys():
        cur_sign = torch.sign(chafen_weight[k])
        old_sign = torch.sign(chafen_model_weight[k])
        sign = cur_sign * old_sign
        sign[sign < 0] = 0
        sign_sum += torch.sum(sign)
        sign_size += sign.numel()

    e = sign_sum / (sign_size + 0.000001)
    print(e)

    
    if e<r:
        V2.load_state_dict(V1_weight)
        c+=1
    return V2.state_dict(),c