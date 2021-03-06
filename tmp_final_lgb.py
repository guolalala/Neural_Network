import lightgbm as lgb
import pandas as pd
import numpy as np
from category_encoders.target_encoder import TargetEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import auc, roc_auc_score, roc_curve
from lightgbm import LGBMRegressor
# 导入数据
train = pd.read_csv('./data/train.csv', index_col='id')
test = pd.read_csv('./data/test.csv', index_col='id')
target = train.pop('fraud_flag')
test = test[train.columns]
# 非数值列
s = train.apply(lambda x:x.dtype)
tecols = s[s=='object'].index.tolist()
print(tecols)


# 模型
def makelgb():
    lgbr = LGBMRegressor(num_leaves=30
                        ,max_depth=5
                        ,learning_rate=.02
                        ,n_estimators=1000
                        ,subsample_for_bin=5000
                        ,min_child_samples=200
                        ,colsample_bytree=.2
                        ,reg_alpha=.1
                        ,reg_lambda=.1
                        )
    return lgbr

# 本地验证
kf = KFold(n_splits=10, shuffle=True, random_state=100)
devscore = []
for tidx, didx in kf.split(train.index):
    tf = train.iloc[tidx]
    df = train.iloc[didx]
    tt = target.iloc[tidx]
    dt = target.iloc[didx]
    te = TargetEncoder(cols=tecols)
    tf = te.fit_transform(tf, tt)
    df = te.transform(df)
    lgbr = makelgb()
    lgbr.fit(tf, tt)
    pre = lgbr.predict(df)
    #print(type(dt), type(pre))
    threshold_rate  = 0.68
    pre= np.where(pre > np.quantile(pre, threshold_rate), 1, 0)
    fpr, tpr, thresholds = roc_curve(dt, pre)
    score = auc(fpr, tpr)
    print(score)
    devscore.append(score)
print("Average Score")
print(np.mean(devscore))
