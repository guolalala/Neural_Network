'''
author: Bodan Chen
Date: 2022-04-14 17:07:06
LastEditors: Bodan Chen
LastEditTime: 2022-04-17 22:20:10
Email: 18377475@buaa.edu.cn
'''
import warnings
warnings.filterwarnings('ignore')
import numpy as np
# 加载莺尾花数据集
from sklearn import datasets
# 导入高斯朴素贝叶斯分类器
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

X, y = datasets.load_iris(return_X_y=True)
print(X)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 使用高斯朴素贝叶斯进行计算
clf = GaussianNB(var_smoothing=1e-8)
clf.fit(X_train, y_train)