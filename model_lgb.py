'''
author: Bodan Chen
Date: 2022-04-17 16:35:31
LastEditors: Bodan Chen
LastEditTime: 2022-04-18 00:24:21
Email: 18377475@buaa.edu.cn
'''
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
data_test=pd.read_csv('./data/test.csv')
data_test=data_test.replace(-1,np.nan)
data_train=reduce_mem_usage(data_train)
data_test=reduce_mem_usage(data_test)
print(type(data_train))

from sklearn.model_selection import KFold
# 分离数据集，方便进行交叉验证
# id,fraud_flag
X_train = data_train.drop(['id','fraud_flag'],axis=1)
#X_test = data_test.drop(['id','fraud_flag'],axis=1)
y_train = data_train.loc[:,'fraud_flag']
# X_train = data.loc[data['sample']=='train', :].drop(['id','issueDate','isDefault', 'sample'], axis=1)
# X_test = data.loc[data['sample']=='test', :].drop(['id','issueDate','isDefault', 'sample'], axis=1)
# y_train = data.loc[data['sample']=='train', 'isDefault']

# 10折交叉验证
folds = 10
seed = 2020
kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

"""对训练集数据进行划分，分成训练集和验证集，并进行相应的操作"""
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score
#pip3 install Booster && pip3 install Dataset && pip3 install Sequence && pip3 install register_logger
# import lightgbm as lgb
# # 数据集划分
# X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2)
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

# """使用训练集数据进行模型训练"""
# model = lgb.train(params, train_set=train_matrix, valid_sets=valid_matrix, num_boost_round=20000, verbose_eval=1000, early_stopping_rounds=200)

# from sklearn import metrics
# from sklearn.metrics import roc_auc_score

# """预测并计算roc的相关指标"""
# val_pre_lgb = model.predict(X_val, num_iteration=model.best_iteration)
# fpr, tpr, threshold = metrics.roc_curve(y_val, val_pre_lgb)
# roc_auc = metrics.auc(fpr, tpr)
# print('未调参前lightgbm单模型在验证集上的AUC:{}'.format(roc_auc))
# """画出roc曲线图"""
# plt.figure(figsize=(8, 8))
# plt.title('Validation ROC')
# plt.plot(fpr, tpr, 'b', label = 'Val AUC = %0.4f' % roc_auc)
# plt.ylim(0,1)
# plt.xlim(0,1)
# plt.legend(loc='best')
# plt.title('ROC')
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# # 画出对角线
# plt.plot([0,1],[0,1],'r--')
# plt.show()

import lightgbm as lgb
"""使用lightgbm 10折交叉验证进行建模预测"""
cv_scores = []
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
    
    cv_scores.append(roc_auc_score(y_val, val_pred))
    print(cv_scores)

print("lgb_scotrainre_list:{}".format(cv_scores))
print("lgb_score_mean:{}".format(np.mean(cv_scores)))
print("lgb_score_std:{}".format(np.std(cv_scores)))

"""保存模型到本地"""
# 保存模型
import pickle
pickle.dump(model, open('data/model_lgb.pkl', 'wb'))