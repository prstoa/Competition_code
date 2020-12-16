#!/usr/bin/env python
# coding: utf-8

# In[9]:


from preprocess.library import *
from sklearn.externals import joblib
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.preprocessing import RobustScaler

# load data
feb_quiz = pd.read_csv('../raw/cst_feat_feb_quiz.csv')
quiz = pd.read_csv('../raw/quiz.csv')

# load model
ovr_rs_lgbm_model1 = joblib.load('../model/ovr_rs_lgbm_model1.pkl')
ovr_rs_lgbm_model2 = joblib.load('../model/ovr_rs_lgbm_model2.pkl')
ovr_rs_lgbm_model3 = joblib.load('../model/ovr_rs_lgbm_model3.pkl')
ovr_rs_lgbm_model4 = joblib.load('../model/ovr_rs_lgbm_model4.pkl')
ovr_rs_lgbm_model5 = joblib.load('../model/ovr_rs_lgbm_model5.pkl')

ovr_none_cat_model1 = joblib.load('../model/ovr_none_cat_model1.pkl')
ovr_none_cat_model2 = joblib.load('../model/ovr_none_cat_model2.pkl')
ovr_none_cat_model3 = joblib.load('../model/ovr_none_cat_model3.pkl')
ovr_none_cat_model4 = joblib.load('../model/ovr_none_cat_model4.pkl')
ovr_none_cat_model5 = joblib.load('../model/ovr_none_cat_model5.pkl')
# Robust Scaling
rs=RobustScaler()
rs_test=scaling(feb_quiz.iloc[:,1:], rs)

# predict
test = feb_quiz.iloc[:,1:]

ovr_cat_predict=0
ovr_cat_predict+=ovr_none_cat_model1.predict_proba(test)/5
ovr_cat_predict+=ovr_none_cat_model2.predict_proba(test)/5
ovr_cat_predict+=ovr_none_cat_model3.predict_proba(test)/5
ovr_cat_predict+=ovr_none_cat_model4.predict_proba(test)/5
ovr_cat_predict+=ovr_none_cat_model5.predict_proba(test)/5

ovr_lgbm_rs_predict=0
ovr_lgbm_rs_predict+=ovr_rs_lgbm_model1.predict_proba(rs_test)/5
ovr_lgbm_rs_predict+=ovr_rs_lgbm_model2.predict_proba(rs_test)/5
ovr_lgbm_rs_predict+=ovr_rs_lgbm_model3.predict_proba(rs_test)/5
ovr_lgbm_rs_predict+=ovr_rs_lgbm_model4.predict_proba(rs_test)/5
ovr_lgbm_rs_predict+=ovr_rs_lgbm_model5.predict_proba(rs_test)/5
# weighted voting
predict = (ovr_cat_predict * 0.5) + (ovr_lgbm_rs_predict * 0.5)

# Put into quiz data
predict=pd.DataFrame(predict)
predict['cst_id_di'] = feb_quiz['cst_id_di']

quiz_cst_id=list(quiz['cst_id_di'])
quiz_mrc_id=list(quiz['MRC_ID_DI'])

predict_list=[]
for i in range(len(quiz)):
    predict_value=predict.set_index('cst_id_di').T[quiz_cst_id[i]][quiz_mrc_id[i]]
    predict_list.append(predict_value)

quiz['Score'] = predict_list

# save quiz
quiz.to_csv('quiz_s.csv', index=False)

