import os
import sys
import numpy as np
from tqdm import tqdm

__all__ = ['train_lr', 'get_metrics']

def get_is_pos(ind_scores, ood_scores, order):
    assert order in ["largest2smallest", "smallest2largest"]
    scores = np.concatenate((ind_scores, ood_scores))
    is_pos = np.concatenate((np.ones(len(ind_scores), dtype="bool"), np.zeros(len(ood_scores), dtype="bool")))
    
    # shuffle before sort
    random_idx = np.random.permutation(list(range(len(scores))))
    scores = scores[random_idx]
    is_pos = is_pos[random_idx]

    idxs = scores.argsort()
    if order == "largest2smallest":
        idxs = np.flip(idxs)
    is_pos = is_pos[idxs]
    return is_pos

def roc(ind_scores, ood_scores, order):
    assert order in ["largest2smallest", "smallest2largest"]
    is_pos = get_is_pos(ind_scores, ood_scores, order)
    TP = 0
    FP = 0
    P = len(ind_scores)
    N = len(ood_scores)
    roc_curve = [[0, 0]]
    for _is_pos in tqdm(is_pos):
        if _is_pos:
            TP += 1
        else:
            FP += 1
        recall = TP / P
        FPR = FP / N
        roc_curve.append([FPR, recall])
    return roc_curve    


def auroc(ind_scores, ood_scores, order):
    assert order in ["largest2smallest", "smallest2largest"]
    roc_curve = roc(ind_scores, ood_scores, order)
    roc_curve = np.array(roc_curve)
    x = roc_curve[:, 0]
    y = roc_curve[:, 1]
    x1 = x[:-1]
    x2 = x[1:]
    y1 = y[:-1]
    y2 = y[1:]
    auc = sum((x2 - x1) * (y1 + y2) / 2)
    return auc

def fpr_at_tpr(ind_scores, ood_scores, order, tpr = 0.95):
    assert order in ["largest2smallest", "smallest2largest"]
    is_pos = get_is_pos(ind_scores, ood_scores, order)
    P = len(ind_scores)
    N = len(ood_scores)
    TP = 0
    FP = 0
    for _is_pos in is_pos:
        if _is_pos:
            TP += 1
        else:
            FP += 1
        TPR = TP / P
        if TPR >= tpr:
            FPR = FP / N
            return FPR

def tnr_at_tpr(ind_scores, ood_scores, order, tpr = 0.95):
    assert order in ["largest2smallest", "smallest2largest"]
    is_pos = get_is_pos(ind_scores, ood_scores, order)
    P = len(ind_scores)
    N = len(ood_scores)
    TP = 0
    TN = N
    for _is_pos in is_pos:
        if _is_pos:
            TP += 1
        else:
            TN -= 1
        TPR = TP / P
        if TPR >= tpr:
            TNR = TN / N
            return TNR

def auin(ind_scores, ood_scores, order):
    assert order in ["largest2smallest", "smallest2largest"]
    is_pos = get_is_pos(ind_scores, ood_scores, order)
    P = len(ind_scores)
    N = len(ood_scores)
    TP = 0
    FP = 0
    recall_prec = []
    for _is_pos in is_pos:
        if _is_pos:
            TP += 1
        else:
            FP += 1
        prec = TP / (TP + FP)
        recall = TP / P
        recall_prec.append([recall, prec])
    recall_prec = np.array(recall_prec)
    x = recall_prec[:,0]
    y = recall_prec[:,1]
    x1 = x[:-1]
    x2 = x[1:]
    y1 = y[:-1]
    y2 = y[1:]
    auin = sum((x2 - x1) * (y1 + y2) / 2)
    return auin
    

def auout(ind_scores, ood_scores, order):
    assert order in ["largest2smallest", "smallest2largest"]
    is_pos = get_is_pos(ind_scores, ood_scores, order)
    is_pos = ~np.flip(is_pos)
    N = len(ind_scores)
    P = len(ood_scores)
    TP = 0
    FP = 0
    recall_prec = []
    for _is_pos in is_pos:
        if _is_pos:
            TP += 1
        else:
            FP += 1
        prec = TP / (TP + FP)
        recall = TP / P
        recall_prec.append([recall, prec])
    recall_prec = np.array(recall_prec)
    x = recall_prec[:,0]
    y = recall_prec[:,1]
    x1 = x[:-1]
    x2 = x[1:]
    y1 = y[:-1]
    y2 = y[1:]
    auout = sum((x2 - x1) * (y1 + y2) / 2)
    return auout

def best_acc(ind_scores, ood_scores, order):
    assert order in ["largest2smallest", "smallest2largest"]
    is_pos = get_is_pos(ind_scores, ood_scores, order)
    P = len(ind_scores)
    N = len(ood_scores)
    TP = 0
    TN = N
    accuracy = 0
    for _is_pos in is_pos:
        if _is_pos:
            TP += 1
        else:
            TN -= 1
        # _acc = (TP+TN) / (P + N)
        _acc = (TP/P + TN/N) / 2
        accuracy = max(accuracy, _acc)
    return accuracy


def train_lr(ind_features_val_for_train, ood_features_val_val_for_train):

    from sklearn.linear_model import LogisticRegressionCV
    train_X = np.concatenate((ind_features_val_for_train, ood_features_val_val_for_train), axis=0)
    train_y = np.concatenate((np.ones(len(ind_features_val_for_train), dtype="int32"), np.zeros(len(ood_features_val_val_for_train), dtype="int32")))
    # lr = LogisticRegressionCV(random_state=42).fit(train_X, train_y)
    lr = LogisticRegressionCV(random_state=42).fit(train_X, train_y)
    return lr
        


def get_metrics(lr, ind_features_test, ood_features_test, acc_type="lr"):
    """
    params:
        ind_features_test: 
            type: numpy.ndarray
            sementic: features for in-distribution dataset
            shape: (N, M); where N = N_sample, M = feature_dim
        ood_features_test:
            similar to ind_features_test
        acc_type:
            methods for computing detection accuracy,
            support "lr" or "best", where "lr" means acc is predicted with logistic regression,
            "best" means acc is computed by enumrate all threshold and return the best.
    return:
        metrics: 
            type: dict
            keys: AUROC, AUIN, AUOUT, DETACC, TNR, FPR
    """
    assert isinstance(ind_features_test, np.ndarray) and isinstance(ood_features_test, np.ndarray)
    assert ind_features_test.shape[1] == ood_features_test.shape[1]
    assert acc_type in ["lr", "best"]

    # print('coefficients:', lr.coef_)
    ind_scores = lr.predict_proba(ind_features_test)[:,1]
    ood_scores = lr.predict_proba(ood_features_test)[:,1]
    
    print("mean ind_scores: {}".format(ind_scores.mean()))
    print("mean ood_scores: {}".format(ood_scores.mean()))
    
    order = "largest2smallest"  # sort score by largest2smallest
    metrics = {}
    metrics['AUROC'] = auroc(ind_scores, ood_scores, order)
    metrics['AUIN'] = auin(ind_scores, ood_scores, order)
    metrics['AUOUT'] = auout(ind_scores, ood_scores, order)
    if acc_type == "lr":
        test_X = np.concatenate((ind_features_test, ood_features_test), axis=0)
        test_y = np.concatenate((np.ones(len(ind_features_test), dtype="int32"), np.zeros(len(ood_features_test), dtype="int32")))
        acc = lr.score(test_X, test_y)
        metrics['DETACC'] = acc
    else:
        metrics['DETACC'] = best_acc(ind_scores, ood_scores, order)
    metrics['TNR@tpr=0.95'] = tnr_at_tpr(ind_scores, ood_scores, order, tpr=0.95)
    metrics['FPR@tpr=0.95'] = fpr_at_tpr(ind_scores, ood_scores, order, tpr=0.95)
    metrics['TNR@tpr=0.8'] = tnr_at_tpr(ind_scores, ood_scores, order, tpr=0.8)
    metrics['FPR@tpr=0.8'] = fpr_at_tpr(ind_scores, ood_scores, order, tpr=0.8)
    return metrics
    
    
def get_metrics_SGD(ind_scores, ood_scores):

    print("mean ind_scores: {}".format(ind_scores.mean()))
    print("mean ood_scores: {}".format(ood_scores.mean()))

    order = "largest2smallest"  # sort score by largest2smallest
    metrics = {}
    metrics['AUROC'] = auroc(ind_scores, ood_scores, order)
    metrics['AUIN'] = auin(ind_scores, ood_scores, order)
    metrics['AUOUT'] = auout(ind_scores, ood_scores, order)
    metrics['DETACC'] = best_acc(ind_scores, ood_scores, order)
    metrics['TNR@tpr=0.95'] = tnr_at_tpr(ind_scores, ood_scores, order, tpr=0.95)
    metrics['FPR@tpr=0.95'] = fpr_at_tpr(ind_scores, ood_scores, order, tpr=0.95)
    metrics['TNR@tpr=0.8'] = tnr_at_tpr(ind_scores, ood_scores, order, tpr=0.8)
    metrics['FPR@tpr=0.8'] = fpr_at_tpr(ind_scores, ood_scores, order, tpr=0.8)
    return metrics
