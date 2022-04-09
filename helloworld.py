'''
author: Bodan Chen
Date: 2022-01-19 16:20:30
LastEditors: Bodan Chen
LastEditTime: 2022-01-28 20:38:08
Email: 18377475@buaa.edu.cn
'''

from cProfile import label
from copy import copy
from turtle import color
from django.urls import translate_url
import numpy as np
import math
from pathlib import Path
import struct
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm_notebook

def tanh(x):
    return np.tanh(x)
def softmax(x):
    x=np.array(x)
    ex=np.exp(x-x.max())
    return ex/ex.sum()
N1=28*28
N2=10
dimentions=[N1,N2]
activation=[tanh,softmax]
distribution=[
    {'b':[0,0]},
    {'b':[0,1],'w':[-math.sqrt(6/(dimentions[0]+dimentions[1])),math.sqrt(6/(dimentions[0]+dimentions[1]))]}
]

def init_parameter_b(layer):
    dist=distribution[layer]['b']
    return np.random.rand(dimentions[layer])*(dist[1]-dist[0])+dist[0]
def init_parameter_w(layer):
    dist=distribution[layer]['w']
    return np.random.rand(dimentions[layer-1],dimentions[layer])*(dist[1]-dist[0])+dist[0]
def init_parameters():
    parameter=[]
    for i in range(len(distribution)):
        layer={}
        for j in distribution[i].keys():
            if j=='b':
                layer['b']=init_parameter_b(i)
            elif j=='w':
                layer['w']=init_parameter_w(i)
        parameter.append(layer)
    return parameter

parameters=init_parameters()

def predict(img,para):
    l0_in=img+para[0]['b']
    l0_out=activation[0](l0_in)
    l1_in=np.dot(l0_out,para[1]['w'])+para[1]['b']
    l1_out=activation[1](l1_in)
    return l1_out
#print(predict(np.random.rand(N1),parameters).argmax())


data_path=Path('./MNIST')
train_img_path=data_path/'train-images-idx3-ubyte'
train_lab_path=data_path/'train-labels-idx1-ubyte'
test_img_path=data_path/'t10k-images-idx3-ubyte'
test_lab_path=data_path/'t10k-images-idx3-ubyte'

train_num=50000
valid_num=10000
test_num=10000
train_f=open(train_img_path,'rb')
struct.unpack('>4i',train_f.read(16))
#print(data_path)
with open(train_img_path,'rb') as f:
    struct.unpack('>4i',f.read(16))
    tmp_img=np.fromfile(f,dtype=np.uint8).reshape(-1,28*28)/255
    train_img=tmp_img[:train_num]
    valid_img=tmp_img[train_num:]
with open(test_img_path,'rb') as f:
    struct.unpack('>4i',f.read(16))
    test_img=np.fromfile(f,dtype=np.uint8).reshape(-1,28*28)/255
with open(train_lab_path,'rb') as f:
    struct.unpack('>2i',f.read(8))
    tmp_lab=np.fromfile(f,dtype=np.uint8)
    train_lab=tmp_lab[:train_num]
    valid_lab=tmp_lab[train_num:]
with open(test_lab_path,'rb') as f:
    struct.unpack('>2i',f.read(8))
    test_lab=np.fromfile(f,dtype=np.uint8)
#print(train_img[0]).reshape(28,28)
img=train_img[0].reshape(28,28)
#plt.imshow(img,cmap='gray')
#a=input()


def d_softmax(data):
    sm=softmax(data)
    return np.diag(sm)-np.outer(sm,sm)
def d_tanh(data):
    return (1/(np.cosh(data))**2)
#print(d_tanh([1,2,3,4]))
differential={softmax:d_softmax,tanh:d_tanh}

onehot=np.identity(dimentions[-1])
def sqr_loss(img,lab,parameter):
    y_pred=predict(img,parameter)
    y=onehot[lab]
    diff=y-y_pred
    return np.dot(diff,diff)
#print(sqr_loss(train_img[0],train_lab[0],parameters))

def grad_parameters(img,lab,parameter):
    l0_in=img+parameter[0]['b']
    l0_out=activation[0](l0_in)
    l1_in=np.dot(l0_out,parameter[1]['w'])+parameter[1]['b']
    l1_out=activation[1](l1_in)

    diff=onehot[lab]-l1_out
    act1=np.dot(differential[activation[1]](l1_in),diff)

    grad_b1=-2*act1
    grad_w1=-2*np.outer(l0_out,act1)
    grad_b0=-2*differential[activation[0]](l0_in)*np.dot(parameter[1]['w'],act1)

    return {'w1':grad_w1,'b1':grad_b1,'b0':grad_b0}
#print(grad_parameters(train_img[2],train_lab[2],init_parameters()))

def valid_loss(parameter):
    loss_accu=0
    for img_i in range(valid_num):
        loss_accu+=sqr_loss(valid_img[img_i],valid_lab[img_i],parameter)
    return loss_accu/(valid_num/10000)
def valid_accuracy(parameter):
    correct=[predict(valid_img[i],parameter).argmax()==valid_lab[i] for i in range(valid_num)]
    return correct.count(True)/len(correct)
def train_loss(parameter):
    loss_accu=0
    for img_i in range(train_num):
        loss_accu+=sqr_loss(train_img[img_i],train_lab[img_i],parameter)
    return loss_accu/(train_num/10000)
def train_accuracy(parameter):
    correct=[predict(train_img[i],parameter).argmax()==train_lab[i] for i in range(train_num)]
    return correct.count(True)/len(correct)
#valid_accuracy(init_parameters())

batch_size=100
def train_batch(current_batch,parameter):
    grad_accu=grad_parameters(train_img[current_batch*batch_size+0],train_lab[current_batch*batch_size+0],parameter)
    for img_i in range(1,batch_size):
        grad_tmp=grad_parameters(train_img[current_batch*batch_size+img_i],train_lab[current_batch*batch_size+img_i],parameter)
        for key in grad_accu.keys():
            grad_accu[key]+=grad_tmp[key]
    for key in grad_accu.keys():
        grad_accu[key]/=batch_size
    return grad_accu
#print(train_batch(0,parameters))
def combine_parameters(parameter,grad,learn_rate):
    parameter_tmp =copy.deepcopy(parameter)
    parameter_tmp[0]['b']-=learn_rate*grad['b0']
    parameter_tmp[1]['b']-=learn_rate*grad['b1']
    parameter_tmp[1]['w']-=learn_rate*grad['w1']
    return parameter_tmp
#print(combine_parameters(parameters,train_batch(0,parameters),1))

parameters=init_parameters()
current_epoch=0
train_loss_list=[]
valid_loss_list=[]
train_accu_list=[]
valid_accu_list=[]

learn_rate=1
epocg_num=5
#print(parameters)

for epoch in tqdm_notebook(range(epocg_num)):
    for i in range(train_num//batch_size):
        if i%100==99:
            print('running batch {}/{}'.format(i+1,train_num//batch_size))
        grad_tmp=train_batch(i,parameters)
        parameters=combine_parameters(parameters,grad_tmp,learn_rate)
    current_epoch+=1
    train_loss_list.append(train_loss(parameters))
    train_accu_list.append(train_accuracy(parameters))
    valid_loss_list.append(valid_loss(parameters))
    valid_accu_list.append(valid_accuracy(parameters))
print("valid accuracy: {}".format(valid_accuracy(parameters)))
print(valid_accu_list)
lower=0
plt.plot(valid_loss_list[lower:],color='black',label='validation loss')
plt.plot(train_loss_list[lower:],color='red',label='train loss')
plt.show()

rand_batch=np.random.randint(train_num//batch_size)
grad_lr=train_batch(rand_batch,parameters)

lr_list=[]
lower=-5
upper=2
step=1
for lr_low in np.linspace(lower,upper,num=(upper-lower)//step+1):
    learn_rate=10**lr_low
    parameters_tmp=combine_parameters(parameters,grad_lr,learn_rate)
    train_loss_tmp=train_loss(parameters_tmp)
    lr_list.append([lr_low,train_loss_tmp])