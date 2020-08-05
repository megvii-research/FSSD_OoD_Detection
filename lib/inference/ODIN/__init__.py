import os
import sys
import pickle

from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from lib.metric import get_metrics, train_lr


def get_mc_ODIN_score(model, dataloader, magnitude, temperature, std):
    criterion = nn.CrossEntropyLoss()
    model.train()
    model = model.cuda()

    mean_scores = []
    var_scores = []

    for data in tqdm(dataloader, desc="get_ODIN_score"):
        if type(data) in [tuple, list] and len(data) == 2:
            imgs, _ = data
        elif isinstance(data, torch.Tensor):
            imgs = data
        else:
            print(type(data))
            raise NotImplementedError

        imgs = imgs.type(torch.FloatTensor).cuda()

        if magnitude > 0:
            imgs.requires_grad = True
            imgs.grad = None
            model.zero_grad()

            logits = model(imgs)
            scaling_logits = logits / temperature
            labels = scaling_logits.data.max(1)[1]

            loss = criterion(scaling_logits, labels)
            loss.backward()
            # Normalizing the gradient to binary in {-1, 1}
            gradient =  torch.ge(imgs.grad.data, 0) # 0 or 1
            gradient = (gradient.float() - 0.5) * 2 # -1 or 1

            if len(std) == 3:
                gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / std[0])
                gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / std[1])
                gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / std[2])
            elif len(std) ==1:
                gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / std[0])

        with torch.no_grad():
            if magnitude > 0:
                imgs_p = torch.add(imgs.data, -magnitude, gradient)
            else:
                imgs_p = imgs

            nsample_logits = []
            nsample_probs = []
            for _ in range(32):
                logits = model(imgs_p)
                logits = logits / temperature
                soft_out = F.softmax(logits, dim=1)
                nsample_logits.append(logits.cpu().numpy())
                nsample_probs.append(soft_out.cpu().numpy())
            nsample_probs = np.mean(nsample_probs, axis=0)
            nsample_vars = np.var(nsample_logits, axis=0)
            # calculate nsample probs
            batch_mean = np.max(nsample_probs, axis=1)
            mean_scores.append(batch_mean)
            # calculate nsample vars
            predicted = np.argmax(nsample_probs, axis=1)
            batch_var = nsample_vars[np.arange(nsample_vars.shape[0]), predicted]
            batch_var = np.array(batch_var)
            var_scores.append(batch_var)
    mean_scores = np.concatenate(mean_scores)
    var_scores = np.concatenate(var_scores)
    assert mean_scores.shape == var_scores.shape,\
        'mean_scores.shape={}, var_scores.shape={}'.format(mean_scores.shape, var_scores.shape)
    return mean_scores, var_scores


def get_ODIN_score(model, dataloader, magnitude, temperature, std):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    model = model.cuda()

    scores = []
    for data in dataloader:
        if type(data) in [tuple, list] and len(data) == 2:
            imgs, _ = data
        elif isinstance(data, torch.Tensor):
            imgs = data
        else:
            print(type(data))
            raise NotImplementedError

        imgs = imgs.type(torch.FloatTensor).cuda()

        if magnitude > 0:
            imgs.requires_grad = True
            imgs.grad = None
            model.zero_grad()

            logits = model(imgs)
            scaling_logits = logits / temperature
            labels = scaling_logits.data.max(1)[1]

            loss = criterion(scaling_logits, labels)
            loss.backward()
            # Normalizing the gradient to binary in {-1, 1}
            gradient =  torch.ge(imgs.grad.data, 0) # 0 or 1
            gradient = (gradient.float() - 0.5) * 2 # -1 or 1

            if len(std) == 3:
                gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / std[0])
                gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / std[1])
                gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / std[2])
            elif len(std) ==1:
                gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / std[0])

        with torch.no_grad():
            if magnitude > 0:
                imgs_p = torch.add(imgs.data, -magnitude, gradient)
            else:
                imgs_p = imgs
            logits = model(imgs_p)
            logits = logits / temperature
            soft_out = F.softmax(logits, dim=1)
            _scores, _ = torch.max(soft_out.data, dim=1)
            scores.append(_scores.cpu().numpy())
    scores = np.concatenate(scores)
    return scores

def search_ODIN_hyperparams(model, 
                            ind_dataloader_val_for_train, 
                            ood_dataloader_val_for_train, 
                            ind_dataloader_val_for_test, 
                            ood_dataloader_val_for_test, 
                            std):
    # magnitude_list = [0, 0.0005, 0.001, 0.0012, 0.0014, 0.0018, 0.002, 0.0024, 0.005, 0.008, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    magnitude_list = [0.2, 0.18, 0.16, 0.14, 0.12, 0.1, 0.08, 0.06, 0.04, 0.02, 0.01, 0.008, 0.006, 0.004, 0.002, 0.001, 0.0008, 0.0006, 0.0004, 0.0002, 0.0001, 0.0]
    # magnitude_list = [0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015]
    # magnitude_list = [0,]
    # magnitude_list = [0, 0.01,]
    temperature_list = [1, 10, 100, 1000]
    # temperature_list = [1000,]
    # temperature_list = [1,]

    best_magnitude = None
    best_temperature = None
    best_tnr = 0
    for m in tqdm(magnitude_list, desc="magnitude"):
        for t in tqdm(temperature_list, desc="temperature"):
            print("get_ODIN_score for ind_scores_train")
            ind_scores_train = get_ODIN_score(model, ind_dataloader_val_for_train, m, t, std)
            print("get_ODIN_score for ood_scores_train")
            ood_scores_train = get_ODIN_score(model, ood_dataloader_val_for_train, m, t, std)
            ind_features_val_for_train = ind_scores_train.reshape(-1,1)
            ood_features_val_for_train = ood_scores_train.reshape(-1,1)
            print("train lr")
            lr = train_lr(ind_features_val_for_train, ood_features_val_for_train)

            print("get_ODIN_score for ind_scores_test")
            ind_scores_test = get_ODIN_score(model, ind_dataloader_val_for_test, m, t, std)
            print("get_ODIN_score for ood_scores_test")
            ood_scores_test = get_ODIN_score(model, ood_dataloader_val_for_test, m, t, std)
            ind_features_val_for_test = ind_scores_test.reshape(-1,1)
            ood_features_val_for_test = ood_scores_test.reshape(-1,1)
            metrics = get_metrics(lr, ind_features_val_for_test, ood_features_val_for_test)
            print("t:{}, m:{}, metrics:{}".format(t, m, metrics))
            if metrics['TNR@tpr=0.95'] > best_tnr:
                best_tnr = metrics['TNR@tpr=0.95']
                best_magnitude = m
                best_temperature = t 
    return best_temperature, best_magnitude

# metrics:  {'FPR@tpr=0.8': 0.6971111111111111, 'AUOUT': 0.5928354030508427, 'AUROC': 0.6206213703703678, 'AUIN': 0.6676188716165599, 'TNR@tpr=0.8': 0.3028888888888889, 'TNR@tpr=0.95': 0.10188888888888889, 'FPR@tpr=0.95': 0.8981111111111111, 'DETACC': 0.6033888888888889}
