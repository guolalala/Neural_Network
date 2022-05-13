'''
author: Bodan Chen
Date: 2022-05-01 22:23:09
LastEditors: Bodan Chen
LastEditTime: 2022-05-10 20:24:35
Email: 18377475@buaa.edu.cn
'''


import numpy as np
import sys
import os
#print(os.getcwd())
import pandas as pd
from sqlalchemy import true
sys.path.insert(0,os.getcwd()+'/GraphEmbedding')
from ge.classify import read_node_label, Classifier
from ge import Node2Vec
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE


def evaluate_embeddings_lgb(embeddings):
    data_train = pd.read_csv('./data/train.csv')
    data_train = data_train.replace(-1,np.nan)
    from sklearn.model_selection import KFold
    # 分离数据集，方便进行交叉验证
    # id,fraud_flag
    X_train = embeddings
    #X_test = data_test.drop(['id','fraud_flag'],axis=1)
    y_train = data_train.loc[:,'fraud_flag']
    #print(type(X_train))
    #print(X_train.index,X_train.columns,len(y_train))
    # 10折交叉验证
    folds = 10
    seed = 2020
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

    """对训练集数据进行划分，分成训练集和验证集，并进行相应的操作"""
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    from sklearn.metrics import roc_auc_score
    import lightgbm as lgb
    """使用lightgbm 10折交叉验证进行建模预测"""
    cv_scores = []
    auc_score=[]
    thre_score=[]
    for i, (train_index, valid_index) in enumerate(kf.split(X_train, y_train)):
        print('************************************ {} ************************************'.format(str(i+1)))
        X_train_split, y_train_split, X_val, y_val = X_train.iloc[train_index], y_train[train_index], X_train.iloc[valid_index], y_train[valid_index]
        
        train_matrix = lgb.Dataset(X_train_split, label=y_train_split)
        valid_matrix = lgb.Dataset(X_val, label=y_val)
        params = {
                    'boosting_type': 'gbdt',
                    'objective': 'binary',
                    'learning_rate': 0.1,
                    'metric': 'auc',
            
                    'min_child_weight': 1e-3,
                    'num_leaves': 31,
                    'max_depth': -1,
                    'reg_lambda': 0,
                    'reg_alpha': 0,
                    'feature_fraction': 1,
                    'bagging_fraction': 1,
                    'bagging_freq': 0,
                    'seed': 2020,
                    'nthread': 8,
                    'silent': True,
                    'verbose': -1,
        }
        model = lgb.train(params, train_set=train_matrix, num_boost_round=20000, valid_sets=valid_matrix, verbose_eval=1000, early_stopping_rounds=200)
        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        print('val_pred=',type(val_pred),len(val_pred),val_pred[:10])
        #val_pred=[list(x).index(max(x)) for x in val_pred]
        #print('val_pred=',type(val_pred),len(val_pred),val_pred)

        print("y_val=",y_val[:50])
        print("y_pred=",type(val_pred),val_pred[:50])
        
        aucmax=[]
        thremax=[]
        for i in range(100):
            def bina(x):
                if x>i/100:
                    return 1
                return 0
            #val_pred0=[bina(x) for x in val_pred]
            val_pred0=np.where(val_pred>=i/100,0,1)
            
            print('val_pred=',type(val_pred0),len(val_pred0),val_pred0[:10])
            
            tmpauc=roc_auc_score(y_val,val_pred0)
            aucmax.append(tmpauc)
            thremax.append(i/100)
        auc_score.append(aucmax)
        thre_score.append(thremax)
        #print('aucmax=%f,thremax=%f'%(aucmax,thremax))
        

        # def binary(x):
        #     if x>0.68:
        #         return 1
        #     return 0
        # val_pred=[binary(x) for x in val_pred]
        # print('val_pred=',type(val_pred),len(val_pred),val_pred[:10])
        # cv_scores.append(roc_auc_score(y_val, val_pred))
        # print(cv_scores)
        
        cv_scores.append(aucmax[50])
        print(cv_scores)
    auc_score=np.array(auc_score)
    maxauc=0
    maxthre=0
    for i in range(100):
        tmpauc=auc_score[:,i]
        print("i=%d lgb_auc_score_mean:%f"%(i,np.mean(tmpauc)))
        if(np.mean(tmpauc)>maxauc):
            maxauc=np.mean(tmpauc)
            maxthre=i/100
    print("maxauc=%f maxthre=%f"%(maxauc,maxthre))
    print("lgb_scotrainre_list:{}".format(cv_scores))
    print("lgb_score_mean:{}".format(np.mean(cv_scores)))
    print("lgb_score_std:{}".format(np.std(cv_scores)))



def evaluate_embeddings(embeddings):
    X, Y = read_node_label('./data/train_labels.txt')
    tr_frac = 0.8
    print("Training classifier using {:.2f}% nodes...".format(
        tr_frac * 100))
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, tr_frac)


def plot_embeddings(embeddings,):
    X, Y = read_node_label('./data/train_labels.txt')

    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    
    data=pd.read_csv('data/adjmatrix.csv',index_col=0)
    tr=pd.read_csv('./data/train.csv',index_col='id')
    inde=tr.index.values
    data=data.loc[inde,inde]
    G=nx.Graph(data.values)
    print(nx.info(G))

    model=Node2Vec(G,walk_length=10,num_walks=80,p=0.25,q=4,workers=1,use_rejection_sampling=0)
    model.train(embed_size=64,window_size=5,iter=3)
    embeddings=model.get_embeddings()
    print(type(embeddings))
    fp='data/embeddings.csv'
    df=pd.DataFrame(embeddings)
    df_T=pd.DataFrame(df.values.T,index=df.columns,columns=df.index)
    print(df_T)
    df_T.to_csv(fp,encoding='utf-8',sep=',',header=true,na_rep=-1,index=true)
    
    
    #print(list(embeddings.items())[:10])

    #evaluate_embeddings(embeddings)
    #plot_embeddings(embeddings)
    embeddings=pd.read_csv('./data/embeddings.csv')
    print(embeddings.head())
    print(embeddings.head().iloc[:,1:])
    #evaluate_embeddings_lgb(embeddings.iloc[:,1:])
    #将邻接矩阵转成edgelist
    # G=nx.from_numpy_matrix('data/adjmatrix.csv',create_using=nx.Graph())
    # print(G)
    #G=nx.read_edgelist('../data/wiki/Wiki_edgelist.txt',
    #                     create_using = nx.DiGraph(), nodetype = None, data = [('weight', int)])
    # model = Node2Vec(G, walk_length=10, num_walks=80,
    #                  p=0.25, q=4, workers=1, use_rejection_sampling=0)
    # model.train(embed_size=64, window_size = 5, iter = 3)
    # embeddings=model.get_embeddings()

    # evaluate_embeddings(embeddings)
    # plot_embeddings(embeddings)
