import numpy as np
import sys
import os
#print(os.getcwd())
import pandas as pd
sys.path.insert(0,os.getcwd()+'/GraphEmbedding')

from sklearn.linear_model import LogisticRegression
import networkx as nx

# 导入数据
train = pd.read_csv('./data/train.csv', index_col='id')
print(train.shape)
test = pd.read_csv('./data/test.csv', index_col='id')
target = train.pop('fraud_flag')

test = test[train.columns]

'''
# 舍弃缺失率高的特征
miss_fearures_dict={'zx_account_status':0.9964717741935484,'loan_is_black':0.9962197580645161,'zx_is_credictcard_current_ovd':0.9952116935483871,'zx_is_current_ovd':0.9873991935483871,'zx_is_lian3_lei6':0.9662298387096774,'als_d15_id_nbank_oth_allnum':0.9027217741935484,'als_m12_id_nbank_finlea_allnum':0.8588709677419355}
miss_fearures=miss_fearures_dict.keys()
train=train.drop(miss_fearures,axis=1)
'''

# embeddings
em=pd.read_csv('./data/embeddings.csv',index_col=0)
em.index=train.index.values

op=2
if op==0:#原特征
    pass
elif op==1:#node2vec裸特征
    train=em
elif op==2:#concat特征
    train=pd.concat([train,em],axis=1)
train=pd.concat([train,target],axis=1)
print(train.head())



i=0
fpath='graphSAGE-pytorch/cora/fraud.content'
with open(fpath,'w') as fp:
    for line in train.values:
        if i==0:
            print(line)
        len_line=len(line)
        name=train.index[i]
        print(name,end='\t',file=fp)
        for num in range(len_line-1):
            print(float(line[num]),end='\t',file=fp)
        print(int(line[-1]),file=fp)
        i+=1
        
#t_test=np.array([1,2,3])
# fp=open('data/train_lables.txt','w')
# for i in y_train:
#     pass
#     print(i,file=fp)

data=pd.read_csv('data/adjmatrix.csv',index_col=0)
tr=pd.read_csv('./data/train.csv',index_col='id')
inde=tr.index.values
data=data.loc[inde,inde]

print(data.tail())

rnum=len(data.index.values)
print(rnum)
cnum=len(data.columns.values)

# with open('graphSAGE-pytorch/cora/miss.csv','w') as fp:
#     for item in data.index.values:
#         if data.loc[item,:].sum()==0:
#             print(item,file=fp)

# fpath='graphSAGE-pytorch/cora/fraud.cites'
# with open(fpath,'a+') as fp:
#     for i in range(rnum):
#         for j in range(i+1):
#             if data.iat[i,j]==1:
#                 print(data.index[i],data.columns[j],file=fp)
   


#G=nx.Graph(data.values)
#print(nx.info(G))