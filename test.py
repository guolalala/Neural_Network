'''
author: Bodan Chen
Date: 2022-02-16 16:30:29
LastEditors: Bodan Chen
LastEditTime: 2022-04-18 10:27:24
Email: 18377475@buaa.edu.cn
'''
# #Code: https://github.com/MorvanZhou/PyTorch-Tutorial
# import torch
# import numpy as np
# 导入数据读取模块
import pandas as pd
import numpy as np

# # details about math operation in torch can be found in: http://pytorch.org/docs/torch.html#math-operations
# np_data=np.arange(6)
# torch_data = torch.from_numpy(np_data)
# tensor2array=torch_data.numpy()
# print(
#     '\nnum array:',np_data,
#     '\ntorch tensor:',torch_data,
#     '\ntensor to array:',tensor2array,
# )

# # np.abs(data)
# # torch.abs(data)

import pickle
f=open('./data/model_lgb.pkl','rb')
data=pickle.load(f)
print(data)

print("****************下载文件中。。。")
#!wget http://tianchi-media.oss-cn-beijing.aliyuncs.com/dragonball/DL/other/data/Emotion_Recognition_File.zip
print("****************下载完成。。。")

df=pd.DataFrame({'A':[1,0,-1],'B':[1,1,1],'C':[-1,-1,-1],'D':[0,0,0]})
print(df)
df=df.replace(-1,np.nan)
print(df)
#df=df.where((pd.notnull(df)),None)
print(df)
print(f'There are {df.isnull().any().sum()} columns in train dataset with missing values.')