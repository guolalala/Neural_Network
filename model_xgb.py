'''
author: Bodan Chen
Date: 2022-04-26 16:23:21
LastEditors: Bodan Chen
LastEditTime: 2022-05-07 20:08:00
Email: 18377475@buaa.edu.cn
'''
from cProfile import label
from pydoc import describe
from tkinter.tix import InputOnly
import pandas as pd
import numpy as np
import warnings
import os
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

"""
sns 相关设置
@return:
"""
# 声明使用 Seaborn 样式
sns.set()
# 有五种seaborn的绘图风格，它们分别是：darkgrid, whitegrid, dark, white, ticks。默认的主题是darkgrid。
sns.set_style("whitegrid")
# 有四个预置的环境，按大小从小到大排列分别为：paper, notebook, talk, poster。其中，notebook是默认的。
sns.set_context('talk')
# 中文字体设置-黑体
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决保存图像是负号'-'显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False
# 解决Seaborn中文显示问题并调整字体大小
sns.set(font='SimHei')

# reduce_mem_usage 函数通过调整数据类型，帮助我们减少数据在内存中占用的空间
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() 
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() 
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

data_train = pd.read_csv('./data/train.csv')
data_train = data_train.replace(-1,np.nan)
#print(data_train.loc[0])
data_test=pd.read_csv('./data/test.csv')
data_test=data_test.replace(-1,np.nan)
data_train=reduce_mem_usage(data_train)
data_test=reduce_mem_usage(data_test)


from sklearn.model_selection import KFold
# 分离数据集，方便进行交叉验证
# id,fraud_flag
X_train = data_train.drop(['id','fraud_flag'],axis=1)
#print(X_train.info())
y_train = data_train.loc[:,'fraud_flag']
#print(y_train.info())

# 10折交叉验证
folds = 10
seed = 2020
kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

"""对训练集数据进行划分，分成训练集和验证集，并进行相应的操作"""
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score


'''xgboost'''
import pickle
import xgboost as xgb
import lightgbm as lgb
"""使用xgboost 10折交叉验证进行建模预测"""
cv_scores = []
val_score=[]
y_score=[]
for i, (train_index, valid_index) in enumerate(kf.split(X_train, y_train)):
    print('************************************ {} ************************************'.format(str(i+1)))
    X_train_split, y_train_split, X_val, y_val = X_train.iloc[train_index], y_train[train_index], X_train.iloc[valid_index], y_train[valid_index]
    
    dtrain=xgb.DMatrix(X_train_split,label=y_train_split)
    dtest=xgb.DMatrix(X_val,label=y_val)

    ''' paramet setting'''
    params={
        'max_depth':2,
        'eta':1,
        'silent':1,
        'objective':'binary:logistic'
    }
    watch_list=[(dtest,'eval'),(dtrain,'train')]
    num_round=5
    model=xgb.train(params=params,dtrain=dtrain,num_boost_round=num_round,evals=watch_list)

    '''predict'''
    val_pred=model.predict(dtest)
    y_score.append(y_val)
    val_score.append(val_pred)

    cv_scores.append(roc_auc_score(y_val,val_pred))
    print(cv_scores)
    # train_matrix = lgb.Dataset(X_train_split, label=y_train_split)
    # valid_matrix = lgb.Dataset(X_val, label=y_val)

    # params = {
    #             'boosting_type': 'gbdt',
    #             'objective': 'binary',
    #             'learning_rate': 0.1,
    #             'metric': 'auc',
        
    #             'min_child_weight': 1e-3,
    #             'num_leaves': 31,
    #             'max_depth': -1,
    #             'reg_lambda': 0,
    #             'reg_alpha': 0,
    #             'feature_fraction': 1,
    #             'bagging_fraction': 1,
    #             'bagging_freq': 0,
    #             'seed': 2020,
    #             'nthread': 8,
    #             'silent': True,
    #             'verbose': -1,
    # }
    # m=xgb.train()
    # model = lgb.train(params, train_set=train_matrix, num_boost_round=20000, valid_sets=valid_matrix, verbose_eval=1000, early_stopping_rounds=200)
    # val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    
    # cv_scores.append(roc_auc_score(y_val, val_pred))
    # print(cv_scores)

maxauc=0
maxthre=0
print(val_score[-1])

# find bestAUC
for i in range(100):
    tmpauc=[]
    for j in range(10):
        tmp_val=np.where(val_score[j]>=i/100,1,0)
        #print(type(tmp_val),len(tmp_val))
        #print(type(y_score[j]), len(y_score[j]))
        tmpauc.append(roc_auc_score(y_score[j],tmp_val))
    
    #tmpauc=auc_score[:,i]
    print("i=%d lgb_auc_score_mean:%f"%(i,np.mean(tmpauc)))
    if(np.mean(tmpauc)>maxauc):
        maxauc=np.mean(tmpauc)
        maxthre=i/100
print("maxauc=%f maxthre=%f"%(maxauc,maxthre))
print("xgb_scotrainre_list:{}".format(cv_scores))
print("xgb_score_mean:{}".format(np.mean(cv_scores)))
print("xgb_score_std:{}".format(np.std(cv_scores)))