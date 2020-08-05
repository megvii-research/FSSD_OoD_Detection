import os
import sys
import pickle

import numpy as np
import torch
from torchvision import transforms
from loguru import logger

from lib.utils.exp import (
    get_model,
    get_transform,
    get_mean, 
    get_std,
    get_dataloader,
)
from lib.utils import split_dataloader
import argparse
parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-i','--ind', type=str, help='in distribution dataset', required=True)
parser.add_argument('-o','--ood', type=str, help='out of distribution dataset', required=True)
parser.add_argument('-m','--model_arch', type=str, help='model architecture', required=True)
parser.add_argument('-b','--batch_size', type=int, default=64)
parser.add_argument('--dataroot',type=str, help='datatset stroage directory',default='/data/datasets')
parser.add_argument('--test_oe', action='store_true', help='whether to use model trained with outlier exposure')
args = vars(parser.parse_args())
print(args)

# ----- load pre-trained model -----
model = get_model(args['ind'], args['model_arch'], test_oe=args['test_oe'])

# ----- load dataset -----
transform = get_transform(args['ind'])
std = get_std(args['ind'])
ind_test_loader = get_dataloader(args['ind'], transform, "test",dataroot=args['dataroot'],batch_size=args['batch_size'])
ood_test_loader = get_dataloader(args['ood'], transform, "test",dataroot=args['dataroot'],batch_size=args['batch_size'])
ind_dataloader_val_for_train, ind_dataloader_val_for_test, ind_dataloader_test = split_dataloader(args['ind'], ind_test_loader, [500,500,-1])
ood_dataloader_val_for_train, ood_dataloader_val_for_test, ood_dataloader_test = split_dataloader(args['ood'], ood_test_loader, [500,500,-1])


# ----- Get Maximum Softmax Probability using get_ODIN_score function -----
from lib.inference.ODIN import get_ODIN_score

# No need to search temperature and magnitude for baseline and OE
best_temperature = 1.0
best_magnitude = 0.0

ind_scores_test = get_ODIN_score(model, ind_dataloader_test, best_magnitude, best_temperature, std=std)
ood_scores_test = get_ODIN_score(model, ood_dataloader_test, best_magnitude, best_temperature, std=std)
ind_features_test = ind_scores_test.reshape(-1,1)
ood_features_test = ood_scores_test.reshape(-1,1)[:len(ind_features_test)]
print("ind_features_test shape: {}".format(ind_features_test.shape))
print("ood_features_test shape: {}".format(ood_features_test.shape))

ind_scores_val_for_train = get_ODIN_score(model, ind_dataloader_val_for_train, best_magnitude, best_temperature, std=std)
ood_scores_val_for_train = get_ODIN_score(model, ood_dataloader_val_for_train, best_magnitude, best_temperature, std=std)
ind_features_val_for_train = ind_scores_val_for_train.reshape(-1,1)
ood_features_val_for_train = ood_scores_val_for_train.reshape(-1,1)
print("ind_features_val_for_train shape: {}".format(ind_features_val_for_train.shape))
print("ood_features_val_for_train shape: {}".format(ood_features_val_for_train.shape))

# ----- Train OoD detector using validation data -----
from lib.metric import get_metrics, train_lr
lr = train_lr(ind_features_val_for_train, ood_features_val_for_train)

# ----- Calculating metrics using test data -----
metrics = get_metrics(lr, ind_features_test, ood_features_test, acc_type="best")
print("metrics: ", metrics)
