import random
import numpy as np
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
import operator
from thop import profile
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
from models import CNNCifar,CNNCifar1
import operator
from thop import profile
def avg(a):
    a=np.array(a)
    return np.mean(a)
def my_sample(arr,r):
    arr=arr.tolist()
    select_num = round(r*len(arr))
    indexs = random.sample(range(len(arr)), select_num)
    myList = [arr[i] for i in indexs]
    while round(avg(myList))!=round(avg(arr)):
        indexs = random.sample(range(len(arr)), select_num)
        myList = [arr[i] for i in indexs]
    for i in range(len(arr)):
        if i not in indexs:
            arr[i]=0
    return np.array(arr)

def inference(model,testloader):
    device="cuda"
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    criterion = torch.nn.NLLLoss().to(device)
    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
    accuracy=correct/total
    return accuracy
def my_sample(arr,r):
    arr=arr.tolist()
    select_num = round(r*len(arr))
    indexs = random.sample(range(len(arr)), select_num)
    myList = [arr[i] for i in indexs]
    for i in range(len(arr)):
        if i not in indexs:
            arr[i]=0
    return np.array(arr)
def inference(model,testloader):
    device="cuda"
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    criterion = torch.nn.NLLLoss().to(device)
    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
    accuracy=correct/total
    return accuracy


def main(V1,V2,testloader):
    args = args_parser()
    device="cuda"
    r=0.27

    chafen=CNNCifar(args=args)
    V1.to(device)
    V2.to(device)
    chafen.to(device)

    V1_weight=V1.state_dict()
    V2_weight=V2.state_dict()

    chafen_weight=chafen.state_dict()
    

    i=0
  
    for k in chafen_weight.keys():
        chafen_weight[k]=V2_weight[k]-V1_weight[k]
        a=chafen_weight[k].cpu().numpy()
        shape=a.shape
        a=a.flatten()
        a=my_sample(a,r)
        a=a.reshape(shape)
        chafen_weight[k]=torch.from_numpy(a)
        chafen_weight[k].to(device)

        i+=1
    chafen.load_state_dict(chafen_weight)

    my_sum1=0
    for k in chafen_weight.keys():
        a=chafen_weight[k].cpu().numpy()
        a=a.flatten()
        b=a.tolist()
        my_sum1+=b.count(0.0)



    

    for k in chafen_weight.keys():
        V2_weight[k]=chafen_weight[k].to(device)+V1_weight[k].to(device)
    V2.load_state_dict(V2_weight)
    


    return V2.state_dict(),my_sum1