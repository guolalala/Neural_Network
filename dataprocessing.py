import numpy as np
import sys
import os
#print(os.getcwd())
import pandas as pd
sys.path.insert(0,os.getcwd()+'/GraphEmbedding')
from ge.classify import read_node_label, Classifier
from ge import Node2Vec
from sklearn.linear_model import LogisticRegression
X, Y = read_node_label('./data/train_labels.txt')

data_train = pd.read_csv('./data/train.csv')
data_train = data_train.replace(-1,np.nan)
data_test=pd.read_csv('./data/test.csv')
data_test=data_test.replace(-1,np.nan)
y_train = data_train.loc[:,'fraud_flag']

#t_test=np.array([1,2,3])
fp=open('data/train_lables.txt','w')
for i in y_train:
    print(i,file=fp)