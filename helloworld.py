'''
author: Bodan Chen
Date: 2022-01-19 16:20:30
LastEditors: Bodan Chen
LastEditTime: 2022-01-19 16:20:30
Email: 18377475@buaa.edu.cn
'''

import numpy as np
import math

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
print(predict(np.random.rand(N1),parameters).argmax())