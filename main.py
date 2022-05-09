'''
author: Bodan Chen
Date: 2022-04-15 19:45:39
LastEditors: Bodan Chen
LastEditTime: 2022-05-07 22:34:09
Email: 18377475@buaa.edu.cn
'''
from nis import cat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import warnings
# warnings.filterwarnings('ignore')

# 读取文件
data_train = pd.read_csv('./data/train.csv')
data_train=data_train.replace(-1,np.nan)
data_test_a = pd.read_csv('./data/test.csv')
data_test_a = data_test_a.replace(-1,np.nan)

# chunker = pd.read_csv('./data/train.csv',chunksize=5)
# for item in chunker:
#     print(type(item))
#     print(len(item))

# 查看数据集的样本个数和原始特征纬度
print(data_test_a.shape)
print(data_train.shape)
print(data_train.columns)
# 数据类型
print(data_train.info())

# 一些基本统计量
print(data_train.describe())
#print(data_train.head(3).append(data_train.tail(3)))
# print(data_train.isnull())
# print(data_train.isnull().sum())
# print(len(data_train))
# print((data_train.isnull().sum()/len(data_train)))

# 查看缺失值，在51列中有43列有缺失值
print(f'There are {data_train.isnull().any().sum()} columns in train dataset with missing values.')

have_null_fea_dict = (data_train.isnull().sum()/len(data_train)).to_dict()
#print(have_null_fea_dict)

# 查看缺失特征中缺失率大于50%的特征
fea_null_moreThanhalf={}
for key,value in have_null_fea_dict.items():
    if value>0.5:
        fea_null_moreThanhalf[key]=value
print(fea_null_moreThanhalf)
print()
tuple_list=[(a,b) for b,a in fea_null_moreThanhalf.items()]
tuple_list=sorted(tuple_list)
print(tuple_list)
# nan可视化
missing = data_train.isnull().sum()/len(data_train)
missing = missing[missing>0]
missing.sort_values(inplace=True)
#print(type(missing))
#missing.plot.bar()
#plt.show()

'''Tips: 比赛大杀器lgb模型可以自动处理缺失值,Task4模型会具体学习模型了解模型哦!'''

# 查看训练集测试集中特征属性只有一值的特征
one_value_fea = [col for col in data_train.columns if data_train[col].nunique()<=1]
one_value_fea_test = [col for col in data_test_a.columns if data_test_a[col].nunique()<=1]
#print(one_value_fea)
#print(one_value_fea_test)
print(f'There are {len(one_value_fea)} columns in train dataset with one unique value.')

# 查看特征的数值类型
numerical_fea = list(data_train.select_dtypes(exclude=['object']).columns)
category_fea = list(filter(lambda x: x not in numerical_fea,list(data_train.columns)))
print(category_fea)

#过滤数值型类别特征
def get_numerical_serial_fea(data,feas):
    numerical_serial_fea = []
    numerical_noserial_fea = []
    for fea in feas:
        temp = data[fea].nunique()
        if temp <= 10:
            numerical_noserial_fea.append(fea)
            continue
        numerical_serial_fea.append(fea)
    return numerical_serial_fea,numerical_noserial_fea
numerical_serial_fea,numerical_noserial_fea = get_numerical_serial_fea(data_train,numerical_fea)
# 所有的特征（除了id）都是离散型变量
#print(numerical_serial_fea)
#print(numerical_noserial_fea)

for item in numerical_noserial_fea:
    print(data_train[item].value_counts())

'''
#按照中位数填充数值型特征
data_train[numerical_fea] = data_train[numerical_fea].fillna(data_train[numerical_fea].median())
data_test_a[numerical_fea] = data_test_a[numerical_fea].fillna(data_train[numerical_fea].median())
#按照众数填充类别型特征
data_train[category_fea] = data_train[category_fea].fillna(data_train[category_fea].mode())
data_test_a[category_fea] = data_test_a[category_fea].fillna(data_train[category_fea].mode())
'''