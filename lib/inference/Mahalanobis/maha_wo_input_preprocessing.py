import os
import sys
import pickle

from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from lib.metric import get_metrics, train_lr

def get_Mahalanobis_score(model, dataloader, num_classes, sample_mean, precision, layer_index, std):
    model.eval()
    model = model.cuda()

    scores = []
    for data in tqdm(dataloader, desc="get_Mahalanobis_score"):
        if type(data) in [tuple, list] and len(data) == 2:
            imgs, _ = data
        elif isinstance(data, torch.Tensor):
            imgs = data
        else:
            print(type(data))
            raise NotImplementedError

        imgs = imgs.type(torch.FloatTensor).cuda()
        with torch.no_grad():
            try:
                noise_out_features = model.intermediate_forward(imgs, layer_index)
            except:
                noise_out_features = model.module.intermediate_forward(imgs, layer_index)
            noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
            noise_out_features = torch.mean(noise_out_features, 2)
        noise_gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1,1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1,1)), 1)      

        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
        scores.extend(noise_gaussian_score.cpu().numpy())
    return scores

def get_Mahalanobis_score_ensemble(model, dataloader, feature_dim_list, num_classes, sample_mean, precision, std=[255,255,255]):
    n_layers = len(feature_dim_list)

    scores_list = []
    for layer_id in range(n_layers):
        scores = get_Mahalanobis_score(model, dataloader, num_classes, sample_mean, precision, layer_id, std)
        scores = np.array(scores).reshape(-1,1)
        scores_list.append(scores)
    return np.concatenate(scores_list, axis=1)
