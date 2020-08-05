import os
import sys
import pickle

from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from lib.metric import get_metrics, train_lr

def compute_fss(model, n_layers, img_size, inp_channel):
    model.eval()
    model = model.cuda()
    n = 100
    noise_imgs = np.random.randint(0,255,(n, inp_channel, img_size, img_size),'uint8')
    noise_imgs = torch.Tensor(noise_imgs).cuda()
    noise_imgs = (noise_imgs - 127.5) / 255

    with torch.no_grad():
        try:
            fss = model.mean_feat_list(noise_imgs)
        except:
            fss = model.module.mean_feat_list(noise_imgs)
    return fss

def get_FSS_score_ensem(model, dataloader, fss, layer_indexs):
    model.eval()
    model = model.cuda()

    all_feat_ensem_score = []
    for data in tqdm(dataloader, desc='Calculating FSSD for each layer...'):
        if type(data) in [tuple, list]:
            imgs = data[0]
        else: # tensor
            imgs = data
        imgs = imgs.cuda()
        # getting features from all layers together
        with torch.no_grad():
            try:
                _, feats = model.feature_list(imgs)
            except:
                _, feats = model.module.feature_list(imgs)

            feat_score = []
            for layer_index in layer_indexs:
                _scores = np.linalg.norm((feats[layer_index] - fss[layer_index]).cpu().detach().numpy(), axis=1).reshape(-1,1)
                feat_score.append(_scores)
            feat_ensem_score = np.concatenate(feat_score, axis=1) # (batch_size, layer_num)
        all_feat_ensem_score.extend(feat_ensem_score) # collect all batches
    all_feat_ensem_score = np.array(all_feat_ensem_score)  # (n, layer_num)
    return all_feat_ensem_score


def search_FSS_hyperparams(model, 
                            fss,
                            layer_indexs,
                            ind_dataloader_val_for_train, 
                            ood_dataloader_val_for_train, 
                            ind_dataloader_val_for_test, 
                            ood_dataloader_val_for_test, 
                            std):
    # magnitude_list = [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]
    magnitude_list = [0.2, 0.18, 0.16, 0.14, 0.12, 0.1, 0.08, 0.06, 0.04, 0.02, 0.01, 0.008, 0.006, 0.004, 0.002, 0.001, 0.0008, 0.0006, 0.0004, 0.0002, 0.0001, 0.0]
    best_magnitude = None
    best_tnr = 0
    for m in tqdm(magnitude_list, desc="magnitude"):
        ind_features_val_for_train = get_FSS_score_ensem_process(model, ind_dataloader_val_for_train, fss, layer_indexs, m, std)
        ood_features_val_for_train = get_FSS_score_ensem_process(model, ood_dataloader_val_for_train, fss, layer_indexs, m, std)

        ind_features_val_for_test = get_FSS_score_ensem_process(model, ind_dataloader_val_for_test, fss, layer_indexs, m, std)
        ood_features_val_for_test = get_FSS_score_ensem_process(model, ood_dataloader_val_for_test, fss, layer_indexs, m, std)
        
        lr = train_lr(ind_features_val_for_train, ood_features_val_for_train)
        metrics = get_metrics(lr, ind_features_val_for_test, ood_features_val_for_test, acc_type="best")
        print("m:{}, metrics:{}".format(m, metrics))
        if metrics['TNR@tpr=0.95'] > best_tnr:
            best_tnr = metrics['TNR@tpr=0.95']
            best_magnitude = m
    return best_magnitude

def get_FSS_score_process(model, dataloader, fss, layer_index, magnitude=0, std=1):
    model.eval()
    model = model.cuda()

    scores = []
    for data in dataloader:
        if type(data) in [tuple, list] and len(data) == 2:
            imgs, _ = data
        elif type(data) == list and len(data) == 1 and isinstance(data[0], torch.Tensor):
            imgs = data[0]
        elif isinstance(data, torch.Tensor):
            imgs = data
        else:
            print(type(data))
            raise NotImplementedError

        temp_imgs = imgs
        temp_imgs = temp_imgs.cuda()     
        if magnitude != 0:
            temp_imgs.requires_grad = True
            temp_imgs.grad = None
            model.zero_grad()
            
            try:
                feat = model.intermediate_forward(temp_imgs, layer_index)
            except:
                feat = model.module.intermediate_forward(temp_imgs, layer_index)
            feat = feat.contiguous().view(feat.size(0),-1)
            
            loss = -(feat - fss[layer_index]).norm(p=2, dim=1).mean()
            loss.backward()
            gradient = temp_imgs.grad.data
            gradient = (gradient.float() - gradient.mean()) / gradient.std()
            
            if len(std) ==3:
                gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / std[0])
                gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / std[1])
                gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / std[2])
            elif len(std) == 1:
                gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / std[0])

            # temp_imgs = torch.add(temp_imgs.data, magnitude, torch.sign(gradient))
            temp_imgs = torch.add(temp_imgs.data, magnitude, gradient)
        
        with torch.no_grad():
            try:
                feat = model.intermediate_forward(temp_imgs, layer_index)
            except:
                feat = model.module.intermediate_forward(temp_imgs, layer_index)
            feat = feat.contiguous().view(feat.size(0),-1)
        _scores = np.linalg.norm((feat - fss[layer_index]).cpu().detach().numpy(), axis=1)
        scores.extend(_scores)
    return scores


def get_FSS_score_ensem_process(model, dataloader, fss, layer_indexs, magnitude, std):
    scores_list = []
    for layer_index in layer_indexs:
        scores = get_FSS_score_process(model, dataloader, fss, layer_index, magnitude, std)
        scores = np.array(scores).reshape(-1,1)
        scores_list.append(scores)
    return np.concatenate(scores_list, axis=1)
