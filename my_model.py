# Data processing, metrics and modeling
from statistics import mode
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score, roc_auc_score, f1_score, roc_curve, auc,precision_recall_curve
from sklearn import metrics
# Lgbm
import catboost
from catboost import Pool
import xgboost as xgb
import lightgbm as lgb
import pandas as pd
import numpy as np
from category_encoders.target_encoder import TargetEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import auc, roc_auc_score, roc_curve
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Suppr warning
import warnings
warnings.filterwarnings("ignore")

import itertools
from scipy import interp

# Plots
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline
from matplotlib import rcParams


#Timer
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('Time taken for Modeling: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

# 导入数据
train = pd.read_csv('./data/train.csv', index_col='id')
test = pd.read_csv('./data/test.csv', index_col='id')
target = train.pop('fraud_flag')

test = test[train.columns]

# 舍弃缺失率高的特征
miss_fearures_dict={'zx_account_status':0.9964717741935484,'loan_is_black':0.9962197580645161,'zx_is_credictcard_current_ovd':0.9952116935483871,'zx_is_current_ovd':0.9873991935483871,'zx_is_lian3_lei6':0.9662298387096774,'als_d15_id_nbank_oth_allnum':0.9027217741935484,'als_m12_id_nbank_finlea_allnum':0.8588709677419355}
miss_fearures=miss_fearures_dict.keys()
train=train.drop(miss_fearures,axis=1)

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
print(train.head())

'''
op==0
LGB 0.7122447311780579
XGB 0.711866004724055

op==1
LGB 0.6827250929832877
XGB 0.6656430109712945

op==2
LGB 0.7594287679352432
XGB 0.7397654388349426

'''

# 非数值列
s = train.apply(lambda x:x.dtype)
tecols = s[s=='object'].index.tolist()

miss_fearures_dict={'zx_account_status':0.9964717741935484,'loan_is_black':0.9962197580645161,'zx_is_credictcard_current_ovd':0.9952116935483871,'zx_is_current_ovd':0.9873991935483871,'zx_is_lian3_lei6':0.9662298387096774,'als_d15_id_nbank_oth_allnum':0.9027217741935484,'als_m12_id_nbank_finlea_allnum':0.8588709677419355}
miss_fearures=list(miss_fearures_dict.keys())
#train=train.drop(miss_fearures[:len(miss_fearures)],axis=1)

# 模型
def makelgb():
    lgbr = LGBMRegressor(
        num_leaves=30
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
def makexgb():
    xgbr=XGBRegressor(
        n_estimators=200,
        seed=2022,
        min_child_weight= 18,
        max_depth=3,
        learning_rate= 0.1,
        eval_metric='auc'
    )
    return xgbr
def makecb():
    cbr=CatBoostRegressor(

    )
    return cbr

def gradient_boosting_model(folds, model,thre):
    print(str(model)+' modeling...')
    start_time = timer(None)
    
    plt.rcParams["axes.grid"] = True

    nfold = folds
    kf = KFold(n_splits=nfold, shuffle=True, random_state=100)
    devscore = []
    for tidx, didx in kf.split(train.index):
        tf = train.iloc[tidx]
        df = train.iloc[didx]
        tt = target.iloc[tidx]
        dt = target.iloc[didx]
        te = TargetEncoder(cols=tecols)
        tf = te.fit_transform(tf, tt)
        df = te.transform(df)
        if model=='LGB':
            clf=makelgb()
        if model=='XGB':
            clf=makexgb()
        if model=='CB':
            clf=makecb()
        
        clf.fit(tf, tt)
        pre = clf.predict(df)
        #print(type(dt), type(pre))
        threshold_rate  = thre
        pre= np.where(pre > np.quantile(pre, threshold_rate), 1, 0)
        fpr, tpr, thresholds = roc_curve(dt, pre)
        score = auc(fpr, tpr)
        print(score)
        devscore.append(score)

    print("Average Score")
    print(np.mean(devscore))
    
    # Timer end    
    timer(start_time)
    return np.mean(devscore)

threshold=0.68
#max_lgb_auc=0
max_lgb_thre=0.65
#max_xgb_auc=0
max_xgb_thre=0.61

'''
# find bestAUC
for i in range(1,100):
    print('i=',i)
    tmp_lgb_auc=gradient_boosting_model(10,'LGB',i/100)
    tmp_xgb_auc=gradient_boosting_model(10,'XGB',i/100)
    if tmp_lgb_auc>max_lgb_auc:
        max_lgb_auc=tmp_lgb_auc
        max_lgb_thre=i/100
    if tmp_xgb_auc>max_xgb_auc:
        max_xgb_auc=tmp_xgb_auc
        max_xgb_thre=i/100
print('max_lgb_auc:',max_lgb_auc)
print('max_lgb_thre',max_lgb_thre)
print('max_xgb_auc',max_xgb_auc)
print('max_xgb_thre',max_xgb_thre)
'''


gradient_boosting_model(10, 'LGB',max_lgb_thre)
gradient_boosting_model(10, 'XGB',max_xgb_thre)