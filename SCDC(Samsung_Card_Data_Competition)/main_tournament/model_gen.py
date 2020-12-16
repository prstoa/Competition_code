#!/usr/bin/env python
# coding: utf-8

# In[4]:


from preprocess.library import *
import pandas as pd
from sklearn.externals import joblib
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import RobustScaler
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# load_data
train = pd.read_csv('../preprocess/train_preprocess.csv')
train_y = pd.read_csv('../preprocess/train_target_preprocess.csv')

# robust scaling
rs=RobustScaler()
rs_x_train2=scaling(train, rs)

# modeling
y=train_y

mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=0)
cbc = CatBoostClassifier(learning_rate=0.035, random_state=42, task_type='GPU', eval_metric='AUC', iterations=1500, early_stopping_rounds=50)
lgbm = LGBMClassifier(device='gpu', learning_rate=0.02, colsample_bytree=0.7, subsample=0.7, num_leaves=7, n_estimators=800)

loop = 0
ovr_cat_predict=0
ovr_cat_model = OneVsRestClassifier(cbc)
for train_index, valid_index in mskf.split(train,y) :
    loop += 1
    X_train, X_valid = train.iloc[train_index], train.iloc[valid_index]
    Y_train, Y_valid = y.iloc[train_index], y.iloc[valid_index]
    
    ovr_cat_model.fit(X_train,Y_train)
    # save_model
    save_model_name = 'ovr_none_catboost_model{}.pkl'.format(loop)
    joblib.dump(ovr_cat_model, save_model_name)
    

loop = 0
ovr_lgbm_rs_predict=0
ovr_rs_lgbm_model = OneVsRestClassifier(lgbm)
for train_index, valid_index in mskf.split(rs_x_train2,y) :
    X_train2, X_valid2 = rs_x_train2[train_index], rs_x_train2[valid_index]
    Y_train2, Y_valid2 = y.iloc[train_index], y.iloc[valid_index]
    
    ovr_rs_lgbm_model.fit(X_train2,Y_train2)
    # save_model
    save_model_name = 'ovr_rs_lgbm_model{}.pkl'.format(loop)
    joblib.dump(ovr_rs_lgbm_model, save_model_name)

