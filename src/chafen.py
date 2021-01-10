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
import copy
import scipy.stats
def ReLuFunc(x):
    x = (np.abs(x) + x) / 2.0
    return x
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    将源域数据和目标域数据转化为核矩阵，即上文中的K
    Params:
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul:
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		sum(kernel_val): 多个核矩阵之和
    '''
    n_samples = int(source.size()[0])+int(target.size()[0])# 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
    total = torch.cat([source, target], dim=0)#将source,target按列方向合并
    #将total复制（n+m）份
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    #将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    #求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
    L2_distance = ((total0-total1)**2).sum(2)
    #调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    #以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    #高斯核函数的数学表达式
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    #得到最终的核矩阵
    return sum(kernel_val)#/len(kernel_val)
 
def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul:
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		loss: MMD loss
    '''
    batch_size = int(source.size()[0])#一般默认为源域和目标域的batchsize相同
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    #根据式（3）将核矩阵分成4部分
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss
#剪枝
def cut(x,a):
    y=np.sign(x)
    for i in range(0,len(x)):
        x[i]=abs(x[i])-a
    return y*ReLuFunc(x)

def relevant1(chafen,chafen_global_model):
    my_sum = 0
    my_size = 0

    for name1,params1 in chafen.named_parameters():
        for name2,params2 in chafen_global_model.named_parameters(): 
            if name1==name2:
                a=params1.detach()*params2.detach()
                my_sum=my_sum+torch.sum(torch.sigmoid(a))
                my_size=my_size+a.numel()
    e1 = my_sum /(my_size+0.00000001)
    return e1
def relevant2(new_model,chafen_global_model):
    my_sum = 0
    my_size = 0

    for name1,params1 in new_model.named_parameters():
        for name2,params2 in chafen_global_model.named_parameters(): 
            if name1==name2:
                a=params1.detach()*params2.detach()
                my_sum=my_sum+torch.sum(torch.sigmoid(a))
                my_size=my_size+a.numel()
    e1 = my_sum /(my_size+0.00000001)
    return e1

def redundancy1(chafen,chafen_local_model):
                 
    my_sum = 0
    my_size = 0
    for name1,params1 in chafen.named_parameters():
        for name2,params2 in chafen_local_model.named_parameters(): 
            if name1==name2 :
                a=params1.detach()-params2.detach()
                my_sum=my_sum+torch.sum(a*a)
                my_size=my_size+a.numel()
    e2=my_sum/(my_size+0.00000001)
    return e2
def redundancy2(new_model,chafen_local_model):
                 
    my_sum = 0
    my_size = 0
    for name1,params1 in new_model.named_parameters():
        for name2,params2 in chafen_local_model.named_parameters(): 
            if name1==name2 :
                a=params1.detach()-params2.detach()
                my_sum=my_sum+torch.sum(a*a)
                my_size=my_size+a.numel()
    e2=my_sum/(my_size+0.00000001)
    return e2






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


def main(r,flag,x,global_models,chafen_global_model,V1,V2,trainloader,chafen_local_weights,testloader):
    args = args_parser()
    device="cuda"
    c1=0
    c2=0
    my_sum1=0
    acc1,acc2=0,0
    r1=0.7
    chafen=CNNCifar(args=args)
    chafen_local_model=CNNCifar(args=args)
    V1.to(device)
    V2.to(device)
    chafen.to(device)
    chafen_local_model.to(device)
    chafen_local_model.load_state_dict(chafen_local_weights)

    V1_weight=V1.state_dict()
    V2_weight=V2.state_dict()
    chafen_global_model_weight=chafen_global_model.state_dict()
    chafen_weight=chafen.state_dict()
    new_model=CNNCifar1(args=args)
    new_model.to(device)
    new_weight=new_model.state_dict()

    for k in chafen_weight.keys():
        chafen_weight[k]=V2_weight[k]-V1_weight[k]
    chafen.load_state_dict(chafen_weight)

    e1 = relevant1(chafen,chafen_global_model)
    e2 = redundancy1(chafen,chafen_local_model)
    print('压缩前',e1)
    print('压缩前',e2)


    if x<10:
        if e1.item()<r:
            print("相关性不够")
            V2.load_state_dict(V1_weight)
            c1+=1
            flag=1
            return V2.state_dict(),chafen_local_weights,c1,c2,my_sum1,acc1,acc2,flag
        else:
            print("相关性够")
            flag=0
            V2.load_state_dict(V2_weight)
            return V2.state_dict(),chafen.state_dict(),c1,c2,my_sum1,acc1,acc2,flag

    else:
        my_sum = 0
        my_size = 0
        for i in range(1,10):
            for k in global_models[0].keys():
                a=global_models[x-i][k]-global_models[x-i-1][k]
                my_sum+=torch.sum(a*a)
                my_size+=a.numel()
        e=my_sum/(my_size+0.00000001)
        print(e.item())
        if e1.item()<r:
            print("相关性不够")
            V2.load_state_dict(V1_weight)
            c1+=1
            flag=1
            return V2.state_dict(),chafen_local_weights,c1,c2,my_sum1,acc1,acc2,flag
        elif e2.item()<e.item():
            print("差异性不够")
            chafen_weight=chafen_local_weights
            for k in V2_weight.keys():
                V2_weight[k]=V1_weight[k]+chafen_weight[k]
            V2.load_state_dict(V2_weight)
            c2+=1
            flag=0
            return V2.state_dict(),chafen_local_weights,c1,c2,my_sum1,acc1,acc2,flag
        else:
            print("都满足，压缩")
            acc1=inference(V2,testloader)
            print('更新后 准确率')
            print(acc1)



            chafen_weight['conv1.weight']=V2_weight['conv1.weight']-V1_weight['conv1.weight']
            c1w=chafen_weight['conv1.weight'].view(450)
            c1w=c1w.cpu().numpy()
            c1w=np.abs(c1w)
            c1w=np.sort(c1w)
            a=c1w[int(len(c1w)*r1)]
            chafen_weight['conv1.weight']=V2_weight['conv1.weight']-V1_weight['conv1.weight']
            c1w=chafen_weight['conv1.weight'].view(450)
            c1w=c1w.cpu().numpy()
            c1w=cut(c1w,a)
            chafen_weight['conv1.weight']= torch.from_numpy(c1w)
            chafen_weight['conv1.weight']=chafen_weight['conv1.weight'].view(6,3,5,5)

            chafen_weight['conv1.bias']=V2_weight['conv1.bias']-V1_weight['conv1.bias']
            c1b=chafen_weight['conv1.bias'].cpu().numpy()
            c1b=np.abs(c1b)
            c1b=np.sort(c1b)
            a=c1b[int(len(c1b)*r1)-1]
            c1b=chafen_weight['conv1.bias'].cpu().numpy()
            c1b=cut(c1b,a)
            chafen_weight['conv1.bias']=torch.from_numpy(c1b)

            chafen_weight['conv2.weight']=V2_weight['conv2.weight']-V1_weight['conv2.weight']
            c2w=chafen_weight['conv2.weight'].view(2400)
            c2w=c2w.cpu().numpy()
            c2w=np.abs(c2w)
            c2w=np.sort(c2w)
            a=c2w[int(len(c2w)*r1)]
            chafen_weight['conv2.weight']=V2_weight['conv2.weight']-V1_weight['conv2.weight']
            c2w=chafen_weight['conv2.weight'].view(2400)
            c2w=c2w.cpu().numpy()
            c2w=cut(c2w,a)
            chafen_weight['conv2.weight']= torch.from_numpy(c2w)
            chafen_weight['conv2.weight']=chafen_weight['conv2.weight'].view(16,6,5,5)

            chafen_weight['conv2.bias']=V2_weight['conv2.bias']-V1_weight['conv2.bias']  
            c2b=chafen_weight['conv2.bias'].cpu().numpy()
            c2b=np.abs(c2b)
            c2b=np.sort(c2b)
            a=c2b[int(len(c2b)*r1)-1]
            c2b=chafen_weight['conv2.bias'].cpu().numpy()
            c2b=cut(c2b,a)
            chafen_weight['conv2.bias']=torch.from_numpy(c2b)

            chafen_weight['fc1.weight']=V2_weight['fc1.weight']-V1_weight['fc1.weight']
            fc1w=chafen_weight['fc1.weight'].view(48000)
            fc1w=fc1w.cpu().numpy()
            fc1w=np.abs(fc1w)
            fc1w=np.sort(fc1w)
            a=fc1w[int(len(fc1w)*r1)]
            chafen_weight['fc1.weight']=V2_weight['fc1.weight']-V1_weight['fc1.weight']
            fc1w=chafen_weight['fc1.weight'].view(48000)
            fc1w=fc1w.cpu().numpy()
            fc1w=cut(fc1w,a)
            chafen_weight['fc1.weight']= torch.from_numpy(fc1w)
            chafen_weight['fc1.weight']=chafen_weight['fc1.weight'].view(120,400)          

            chafen_weight['fc1.bias']=V2_weight['fc1.bias']-V1_weight['fc1.bias']
            fc1b=chafen_weight['fc1.bias'].cpu().numpy()
            fc1b=np.abs(fc1b)
            fc1b=np.sort(fc1b)
            a=fc1b[int(len(fc1b)*r1)-1]
            c1b=chafen_weight['fc1.bias'].cpu().numpy()
            c1b=cut(fc1b,a)
            chafen_weight['fc1.bias']=torch.from_numpy(fc1b)

            chafen_weight['fc2.weight']=V2_weight['fc2.weight']-V1_weight['fc2.weight']
            fc2w=chafen_weight['fc2.weight'].view(10080)
            fc2w=fc2w.cpu().numpy()
            fc2w=np.abs(fc2w)
            fc2w=np.sort(fc2w)
            a=fc2w[int(len(fc2w)*r1)]
            chafen_weight['fc2.weight']=V2_weight['fc2.weight']-V1_weight['fc2.weight']
            fc2w=chafen_weight['fc2.weight'].view(10080)
            fc2w=fc2w.cpu().numpy()
            fc2w=cut(fc2w,a)
            chafen_weight['fc2.weight']= torch.from_numpy(fc2w)
            chafen_weight['fc2.weight']=chafen_weight['fc2.weight'].view(84,120)     

            chafen_weight['fc2.bias']=V2_weight['fc2.bias']-V1_weight['fc2.bias']
            fc2b=chafen_weight['fc2.bias'].cpu().numpy()
            fc2b=np.abs(fc2b)
            fc2b=np.sort(fc2b)
            a=fc2b[int(len(fc2b)*r1)-1]
            fc2b=chafen_weight['fc2.bias'].cpu().numpy()
            fc2b=cut(fc2b,a)
            chafen_weight['fc2.bias']=torch.from_numpy(fc2b)

            chafen_weight['fc3.weight']=V2_weight['fc3.weight']-V1_weight['fc3.weight']
            fc3w=chafen_weight['fc3.weight'].view(840)
            fc3w=fc3w.cpu().numpy()
            fc3w=np.abs(fc3w)
            fc3w=np.sort(fc3w)
            a=fc3w[int(len(fc3w)*r1)]
            chafen_weight['fc3.weight']=V2_weight['fc3.weight']-V1_weight['fc3.weight']
            fc3w=chafen_weight['fc3.weight'].view(840)
            fc3w=fc3w.cpu().numpy()
            fc3w=cut(fc3w,a)
            chafen_weight['fc3.weight']= torch.from_numpy(fc3w)
            chafen_weight['fc3.weight']=chafen_weight['fc3.weight'].view(10,84)


            chafen_weight['fc3.bias']=V2_weight['fc3.bias']-V1_weight['fc3.bias']
            fc3b=chafen_weight['fc3.bias'].cpu().numpy()
            fc3b=np.abs(fc3b)
            fc3b=np.sort(fc3b)
            a=fc3b[int(len(fc3b)*r1)-1]
            fc3b=chafen_weight['fc3.bias'].cpu().numpy()
            fc3b=cut(fc3b,a)
            chafen_weight['fc3.bias']=torch.from_numpy(fc3b)


            chafen.load_state_dict(chafen_weight)
            new_e1 = relevant1(chafen,chafen_global_model)
            new_e2 = redundancy1(chafen,chafen_local_model)
            print('剪枝后',new_e1)
            print('剪枝后',new_e2)

            

       
            new_weight['conv11.weight']=V1_weight['conv1.weight']
            new_weight['conv12.weight']=chafen_weight['conv1.weight']
            new_weight['conv11.bias']=V1_weight['conv1.bias']
            new_weight['conv12.bias']=chafen_weight['conv1.bias']

            new_weight['conv21.weight']=V1_weight['conv2.weight']
            new_weight['conv22.weight']=chafen_weight['conv2.weight']
            new_weight['conv21.bias']=V1_weight['conv2.bias']
            new_weight['conv22.bias']=chafen_weight['conv2.bias']
    
            
            new_weight['fc11.weight']=V1_weight['fc1.weight']
            new_weight['fc12.weight']=chafen_weight['fc1.weight']
            new_weight['fc11.bias']=V1_weight['fc1.bias']
            new_weight['fc12.bias']=chafen_weight['fc1.bias']

            new_weight['fc21.weight']=V1_weight['fc2.weight']
            new_weight['fc22.weight']=chafen_weight['fc2.weight']
            new_weight['fc21.bias']=V1_weight['fc2.bias']
            new_weight['fc22.bias']=chafen_weight['fc2.bias']

            new_weight['fc31.weight']=V1_weight['fc3.weight']
            new_weight['fc32.weight']=chafen_weight['fc3.weight']
            new_weight['fc31.bias']=V1_weight['fc3.bias']
            new_weight['fc32.bias']=chafen_weight['fc3.bias']


            new_model.load_state_dict(new_weight)
           
            m=['conv11.weight','conv21.weight','fc11.weight','fc21.weight','fc31.weight','conv11.bias','conv21.bias','fc11.bias','fc21.bias','fc31.bias']
            for name1,param1 in new_model.named_parameters():
                if name1 in m:
                    param1.requires_grad=False
                else:
                    param1.requires_grad=True

            for name,param in V2.named_parameters():
                param.requires_grad=False
            for name,param in V1.named_parameters():
                param.requires_grad=False 
            for name,param in chafen_global_model.named_parameters():
                param.requires_grad=False 
            for name,param in chafen_local_model.named_parameters():
                param.requires_grad=False 
            
            
            optimizer = torch.optim.Adam(new_model.parameters(), lr=0.0001,weight_decay=1e-4)
            criterion1 = torch.nn.NLLLoss().to(device)
            criterion2 = torch.nn.KLDivLoss().to(device)
            epoch_loss = []
            EPS = 1e-8
            locked_masks = {n: torch.abs(w) < EPS for n, w in new_model.named_parameters() if n.endswith('weight')}
            chafen_global_model_weight.requires_grad=False
            chafen_local_weights.requires_grad=False



            for epoch in tqdm(range(50)):
                batch_loss=[]
                batch_loss1=[]
                batch_loss2=[]
                batch_loss3=[]
                batch_loss4=[]
                for batch_idx, (images, labels) in enumerate(trainloader):

                    my_e1,my_size=0.0,0.0
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = new_model(images)
                    loss1 = criterion1(outputs, labels)
                    x=new_model.f(images)
                    y=V2.f(images)
                    z=new_model.f1(images)
                    loss2=0.8*mmd_rbf(x,y)+0.2*mmd_rbf(x,z)
                    loss=0.7*loss2+0.3*loss1
                    loss.backward()
                    for n, w in new_model.named_parameters():                                                                                                                                                                           
                        if w.grad is not None and n in locked_masks:                                                                                                                                                                                   
                            w.grad[locked_masks[n]] = 0
                    optimizer.step()
                    if batch_idx % 50 == 0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch+1, batch_idx * len(images), len(trainloader.dataset),
                            100. * batch_idx / len(trainloader), loss.item()))
                    batch_loss.append(loss.item())
                    batch_loss1.append(loss1.item())
                    batch_loss2.append(loss2.item())



            loss_avg = sum(batch_loss)/len(batch_loss)
            loss_avg1 = sum(batch_loss1)/len(batch_loss1)
            loss_avg2 = sum(batch_loss2)/len(batch_loss2)



            print('\nTrain loss:', loss_avg)
            print('\nTrain loss1:', loss_avg1)
            print('\nTrain loss2:', loss_avg2)

            new_weight=new_model.state_dict()

 
            chafen_weight['conv1.weight']=new_weight['conv12.weight']
            chafen_weight['conv1.bias']=new_weight['conv12.bias']
            chafen_weight['conv2.weight'] =new_weight['conv22.weight']
            chafen_weight['conv2.bias']=new_weight['conv22.bias']
            
            chafen_weight['fc1.weight']=new_weight['fc12.weight']
            chafen_weight['fc1.bias']=new_weight['fc12.bias']
            chafen_weight['fc2.weight'] =new_weight['fc22.weight']
            chafen_weight['fc2.bias']=new_weight['fc22.bias']
            chafen_weight['fc3.weight'] =new_weight['fc32.weight']
            chafen_weight['fc3.bias']=new_weight['fc32.bias']
            chafen.load_state_dict(chafen_weight)

            e1 = relevant1(chafen,chafen_global_model)
            e2 = redundancy1(chafen,chafen_local_model)
            print("retrain后 ",e1)
            print("retrain后 ",e2)


            
            #统计0
                
            for k in chafen_weight.keys():
                a=chafen_weight[k].cpu().numpy()
                a=a.flatten()
                b=a.tolist()
                my_sum1+=b.count(0.0)
            print(my_sum1)


            for k in V2_weight.keys():
                V2_weight[k]=chafen_weight[k]+V1_weight[k]
            V2.load_state_dict(V2_weight)
                
                
            acc2=inference(V2,testloader)
            print('恢复后 准确率')
            print(acc2)

            flag=0
            
            for name,param in V2.named_parameters():
                param.requires_grad=True
            for name,param in V1.named_parameters():
                param.requires_grad=True 
            for name,param in chafen_global_model.named_parameters():
                param.requires_grad=True
            for name,param in chafen_local_model.named_parameters():
                param.requires_grad=True

            return V2.state_dict(),chafen.state_dict(),c1,c2,my_sum1,acc1,acc2,flag