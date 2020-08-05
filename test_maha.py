import os
import sys
import pickle

import numpy as np
import torch
from torchvision import transforms
from loguru import logger
from lib.utils import split_dataloader

from lib.utils.exp import (
    get_model,
    get_transform,
    get_mean, 
    get_std,
    get_dataloader,
    get_img_size,
    get_inp_channel,
)
import argparse
parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-i','--ind', type=str, help='in distribution dataset', required=True)
parser.add_argument('-o','--ood', type=str, help='out of distribution dataset', required=True)
parser.add_argument('-m','--model_arch', type=str, help='model architecture', required=True)
parser.add_argument('-b','--batch_size', type=int, help='batch size', default=512)
parser.add_argument('--dataroot',type=str, help='datatset stroage directory',default='/data/datasets')

args = vars(parser.parse_args())
print(args)

# ----- load pre-trained model -----
model = get_model(args['ind'], args['model_arch'])

# ----- load dataset -----
transform = get_transform(args['ind'])
std = get_std(args['ind'])
img_size = get_img_size(args['ind'])
inp_channel = get_inp_channel(args['ind'])
batch_size = args['batch_size'] # recommend: 64 for ImageNet, CelebA, MS1M

ind_train_loader = get_dataloader(args['ind'], transform, "train",dataroot=args['dataroot'],batch_size=batch_size)
ind_test_loader = get_dataloader(args['ind'], transform, "test", dataroot=args['dataroot'], batch_size=batch_size)
ood_test_loader = get_dataloader(args['ood'], transform, "test", dataroot=args['dataroot'], batch_size=batch_size)
ind_dataloader_val_for_train, ind_dataloader_val_for_test, ind_dataloader_test = split_dataloader(args['ind'], ind_test_loader, [500, 500, -1], random=True)
ood_dataloader_val_for_train, ood_dataloader_val_for_test, ood_dataloader_test = split_dataloader(args['ood'], ood_test_loader, [500,500, -1], random=True)


# if args['ind'] == 'dogs50B':
#     ind_train_loader = get_dataloader('dogs50A', transform, "train",dataroot=args['dataroot'],batch_size=args['batch_size'])

# ---- Calculating Mahanalobis distance ----
from lib.inference import get_feature_dim_list
from lib.inference.Mahalanobis import (
        sample_estimator, 
        search_Mahalanobis_hyperparams,
        get_Mahalanobis_score_ensemble,
    )
from lib.metric import get_metrics, train_lr
from lib.utils import split_dataloader

logger.info("search Maha params")

feature_dim_list, num_classes = get_feature_dim_list(model, img_size, inp_channel, flat=False)
# print('number of classes', num_classes)
# print(feature_dim_list)
sample_mean, precision = sample_estimator(model, num_classes, feature_dim_list, ind_train_loader)

layer_indexs = list(range(len(feature_dim_list)))

best_magnitude = search_Mahalanobis_hyperparams(model, sample_mean, precision, layer_indexs, num_classes,
                                ind_dataloader_val_for_train, 
                                ood_dataloader_val_for_train,
                                ind_dataloader_val_for_test, 
                                ood_dataloader_val_for_test, 
                                std=std)

ind_features_val_for_train = get_Mahalanobis_score_ensemble(model, ind_dataloader_val_for_train, layer_indexs, num_classes, sample_mean, precision, best_magnitude, std=std)
ood_features_val_for_train = get_Mahalanobis_score_ensemble(model, ood_dataloader_val_for_train, layer_indexs, num_classes, sample_mean, precision, best_magnitude, std=std)

ind_features_test = get_Mahalanobis_score_ensemble(model, ind_dataloader_test, layer_indexs, num_classes, sample_mean, precision, best_magnitude, std=std)
ood_features_test = get_Mahalanobis_score_ensemble(model, ood_dataloader_test, layer_indexs, num_classes, sample_mean, precision, best_magnitude, std=std)[:len(ind_features_test)]

# ----- Training OoD detector using validation data -----
lr = train_lr(ind_features_val_for_train, ood_features_val_for_train)

# ----- Calculating metrics using test data -----
metrics = get_metrics(lr, ind_features_test, ood_features_test, acc_type="best")
print("best params: ", best_magnitude)
print("metrics: ", metrics)

