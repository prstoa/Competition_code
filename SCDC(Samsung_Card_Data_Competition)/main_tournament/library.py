#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc
from scipy import interp
import random
from sklearn.neighbors import NearestNeighbors


# In[ ]:


def get_tail_label(df: pd.DataFrame, ql=[0.05, 1.]) -> list:
    """
    Find the underrepresented targets.
    Underrepresented targets are those which are observed less than the median occurance.
    Targets beyond a quantile limit are filtered.
    """
    irlbl = df.sum(axis=0)
    irlbl = irlbl[(irlbl > irlbl.quantile(ql[0])) & ((irlbl < irlbl.quantile(ql[1])))]  # Filtering
    irlbl = irlbl.max() / irlbl
    threshold_irlbl = irlbl.median()
    tail_label = irlbl[irlbl > threshold_irlbl].index.tolist()
    return tail_label

def get_minority_samples(X: pd.DataFrame, y: pd.DataFrame, ql=[0.05, 1.]):
    """
    return
    X_sub: pandas.DataFrame, the feature vector minority dataframe
    y_sub: pandas.DataFrame, the target vector minority dataframe
    """
    tail_labels = get_tail_label(y, ql=ql)
    index = y[y[tail_labels].apply(lambda x: (x == 1).any(), axis=1)].index.tolist()
    
    X_sub = X[X.index.isin(index)].reset_index(drop = True)
    y_sub = y[y.index.isin(index)].reset_index(drop = True)
    return X_sub, y_sub

def nearest_neighbour(X: pd.DataFrame, neigh) -> list:
    """
    Give index of 10 nearest neighbor of all the instance
    
    args
    X: np.array, array whose nearest neighbor has to find
    
    return
    indices: list of list, index of 5 NN of each element in X
    """
    nbs = NearestNeighbors(n_neighbors=neigh, metric='euclidean', algorithm='kd_tree').fit(X)
    euclidean, indices = nbs.kneighbors(X)
    return indices

def MLSMOTE(X, y, n_sample, neigh=5):
    """
    Give the augmented data using MLSMOTE algorithm
    
    args
    X: pandas.DataFrame, input vector DataFrame
    y: pandas.DataFrame, feature vector dataframe
    n_sample: int, number of newly generated sample
    
    return
    new_X: pandas.DataFrame, augmented feature vector data
    target: pandas.DataFrame, augmented target vector data
    """
    indices2 = nearest_neighbour(X, neigh=5)
    n = len(indices2)
    new_X = np.zeros((n_sample, X.shape[1]))
    target = np.zeros((n_sample, y.shape[1]))
    for i in range(n_sample):
        reference = random.randint(0, n-1)
        neighbor = random.choice(indices2[reference, 1:])
        all_point = indices2[reference]
        nn_df = y[y.index.isin(all_point)]
        ser = nn_df.sum(axis = 0, skipna = True)
        target[i] = np.array([1 if val > 0 else 0 for val in ser])
        ratio = random.random()
        gap = X.loc[reference,:] - X.loc[neighbor,:]
        new_X[i] = np.array(X.loc[reference,:] + ratio * gap)
    new_X = pd.DataFrame(new_X, columns=X.columns)
    target = pd.DataFrame(target, columns=y.columns)
    return new_X, target


# In[ ]:


def target_score(model_predict, test_y, target_class):
    y_valid = test_y.iloc[:,1:]
    y_test=y_valid
    y_score=model_predict

    lw = 2
    n_classes = len(target_class)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in target_class:
        fpr[i], tpr[i], _ = roc_curve(y_test.iloc[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(np.array(y_test).ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in target_class]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in target_class:
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # lift score
    len_top20=np.int(np.round(len(model_predict)*0.2))
    top20=pd.DataFrame(model_predict).apply(lambda x: max(x), axis=1).sort_values(ascending=False)[:len_top20]
    pred20=np.array(pd.DataFrame(model_predict).iloc[top20.index]).argmax(axis=1)
    
    # target_lift_score
    target_lift=np.zeros(shape=(11,1))
    for i in target_class:
        target_lift[i]=(len(pred20[pred20==i])/len(pred20))/(len(test_y[test_y.iloc[:,i+1]==1])/len(test_y))
    
    # target_mean_lift_score
    target_mean_lift=0
    for i in target_class:
        target_mean_lift+=(len(pred20[pred20==i])/len(pred20))/(len(test_y[test_y.iloc[:,i+1]==1])/len(test_y))
    target_mean_lift=target_mean_lift/len(target_class)
    
    target_final = (0.7*(target_mean_lift/5)) + (0.3*(roc_auc['micro'] + roc_auc['macro'])/2)
    
    print(roc_auc)
    print('mean(micro, macro)', (roc_auc['micro'] + roc_auc['macro'])/2)
    for i in target_class:
        print('class ',i,'lift :',target_lift[i],'\n')
    print('target_mean_lift :', target_mean_lift)
    print('target_final_score :', target_final)


# In[ ]:


def totality_score(model_predict, test_y):
    y_valid = test_y.iloc[:,1:]
    auroc_macro=roc_auc_score(np.array(y_valid), model_predict,multi_class="ovr", average= 'macro')
    auroc_micro=roc_auc_score(np.array(y_valid), model_predict,multi_class="ovr", average= 'micro')

    len_top20=np.int(np.round(len(model_predict)*0.2))

    top20=pd.DataFrame(model_predict).apply(lambda x: max(x), axis=1).sort_values(ascending=False)[:len_top20]
    pred20=np.array(pd.DataFrame(model_predict).iloc[top20.index]).argmax(axis=1)

    lift_score=0
    for i in range(11):
        lift_score+=(len(pred20[pred20==i])/len(pred20))/(len(test_y[test_y.iloc[:,i+1]==1])/len(test_y))
    lift_score=lift_score/(len(test_y.columns)-1)

    final_score_macro = (0.7*(lift_score/5)) + (0.3*auroc_macro)
    final_score_micro = (0.7*(lift_score/5)) + (0.3*auroc_micro)

    print('auroc_macro :', auroc_macro)
    print('auroc_micro :', auroc_micro)
    print('lift_score :', lift_score)
    print('final_score_micro :', final_score_micro)
    print('final_score_macro :', final_score_macro)


# In[ ]:


def scaling(trans_data, method):
    scaled = method.fit_transform(trans_data)
    return scaled


# In[ ]:


def balancing_auroc_lift(auroc_predict, lift_predict, auroc_rate, lift_rate):
    predict = (auroc_predict * auroc_rate) + (lift_predict * lift_rate)
    return predict

