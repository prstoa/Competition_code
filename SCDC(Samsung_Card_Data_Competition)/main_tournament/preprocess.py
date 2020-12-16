#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from library import *
import time

# load_dataset
train_y=pd.read_csv('../raw/train.csv')
val_y=pd.read_csv('../raw/val.csv')
train_val=pd.read_csv('../raw/cst_feat_jan.csv')

# seperate to training_data and validation_data
# cst_feat_jan.csv 에서 train, validation 분리
train_val_y=pd.concat([train_y,val_y], ignore_index=True)

duplicated=train_val_y[train_val_y.duplicated('cst_id_di')]['cst_id_di'].values

train_id = list(set(train_y['cst_id_di']))
val_id = list(set(val_y['cst_id_di']))
# 과적합 방지를 위해 val.csv파일에서 train.csv와 val.csv에 중복인 id를 제거
val_id = list((set(val_id) - set(train_id)))

train=train_val[train_val['cst_id_di'].apply(lambda x: x in train_id)]
train=train.sort_values(['cst_id_di']).reset_index(drop=True)
train=train.iloc[:,1:]

val=train_val[train_val['cst_id_di'].apply(lambda x: x in val_id)]
val=val.sort_values(['cst_id_di']).reset_index(drop=True)
val=val.iloc[:,1:]

# target data preprocess
# multi-label encoding
y=pd.concat([train_y,val_y], ignore_index=True)
y=pd.get_dummies(y, columns=['MRC_ID_DI'])

du3=list(y['cst_id_di'].value_counts()[y['cst_id_di'].value_counts()==3].index)
du2=list(y['cst_id_di'].value_counts()[y['cst_id_di'].value_counts()==2].index)
du1=list(y['cst_id_di'].value_counts()[y['cst_id_di'].value_counts()==1].index)

df_du1=y.set_index(['cst_id_di']).loc[du1]
df_du2=pd.DataFrame(index=range(0), columns=y.columns).set_index(['cst_id_di'])
for i in range(0,len(du2)*2,2):
    df_du2=df_du2.append(y.set_index(['cst_id_di']).loc[du2].sort_index().iloc[i:i+1,:] + y.set_index(['cst_id_di']).loc[du2].sort_index().iloc[i+1:i+2,:])

df_du3=pd.DataFrame(index=range(0), columns=y.columns).set_index(['cst_id_di'])
for i in range(0,len(du3)*3,3):
    df_du3=df_du3.append(y.set_index(['cst_id_di']).loc[du3].sort_index().iloc[i:i+1,:] + y.set_index(['cst_id_di']).loc[du3].sort_index().iloc[i+1:i+2,:] + y.set_index(['cst_id_di']).loc[du3].sort_index().iloc[i+2:i+3,:])
    
y=df_du1.append(df_du2)
y=y.append(df_du3)
en_train_y=y.loc[train_id].sort_index().reset_index()
en_val_y=y.loc[val_id].sort_index().reset_index()
en_val_y=en_val_y.iloc[:,1:]


# MLSMOTE for training_dataset and encoding_target_dataset
for i in range(10):
    start = time.time()
    X_sub, y_sub = get_minority_samples(train, en_train_y)  # Getting minority samples of that datframe
    X_res, y_res = MLSMOTE(X_sub, y_sub, 40000, 5)  # Applying MLSMOTE to augment the dataframe
    train = pd.concat([train, X_res], axis=0, ignore_index=True)
    en_train_y = pd.concat([en_train_y, y_res], axis=0)
    end = time.time() - start
    print("cycle =",i+1)
    print("걸린 시간 : ", int(end/60),"min  ", int(end % 60) ,"sec")
mlsmote_train = train
mlsmote_y = en_train_y

# save_training_dataset
mlsmote_train.to_csv('../preprocess/train_preprocess.csv',index=False)
val.to_csv('../preprocess/val_preprocess.csv',index=False)

# save_encoding_target_dataset
mlsmote_y.to_csv('../preprocess/train_target_preprocess.csv', index=False)
en_val_y.to_csv('../preprocess/val_target_preprocess.csv', index=False)

