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
parser.add_argument('-b','--batch_size', type=int, help='batch size', default=512)
parser.add_argument('--dataroot',type=str, help='datatset stroage directory', default='/data/datasets')
args = vars(parser.parse_args())
print(args)

# ----- load pre-trained model -----
model = get_model(args['ind'], args['model_arch'])

# ----- load dataset -----
transform = get_transform(args['ind'])
std = get_std(args['ind'])
ind_test_loader = get_dataloader(args['ind'], transform, "test",dataroot=args['dataroot'],batch_size=args['batch_size'])
ood_test_loader = get_dataloader(args['ood'], transform, "test",dataroot=args['dataroot'],batch_size=args['batch_size'])
ind_dataloader_val_for_train, ind_dataloader_val_for_test, ind_dataloader_test = split_dataloader(args['ind'], ind_test_loader, [500,500,-1], random=True)
ood_dataloader_val_for_train, ood_dataloader_val_for_test, ood_dataloader_test = split_dataloader(args['ood'], ood_test_loader, [500,500,-1], random=True)


# ----- Calculate best temperature and magnitude for input pre-processing -----
from lib.inference.ODIN import search_ODIN_hyperparams, get_ODIN_score
logger.info("search ODIN params")
best_temperature, best_magnitude = search_ODIN_hyperparams(model, ind_dataloader_val_for_train, ood_dataloader_val_for_train, ind_dataloader_val_for_test, ood_dataloader_val_for_test, std=std)
print("best params: ", best_temperature, best_magnitude)

# ----- Calculate ODIN score for validation data -----
ind_scores_val_for_train = get_ODIN_score(model, ind_dataloader_val_for_train, best_magnitude, best_temperature, std=std)
ood_scores_val_for_train = get_ODIN_score(model, ood_dataloader_val_for_train, best_magnitude, best_temperature, std=std)
ind_features_val_for_train = ind_scores_val_for_train.reshape(-1,1)
ood_features_val_for_train = ood_scores_val_for_train.reshape(-1,1)

# ----- Calculate ODIN score for test data -----
ind_scores_test = get_ODIN_score(model, ind_dataloader_test, best_magnitude, best_temperature, std=std)
ood_scores_test = get_ODIN_score(model, ood_dataloader_test, best_magnitude, best_temperature, std=std)[:len(ind_scores_test)]
ind_features_test = ind_scores_test.reshape(-1,1)
ood_features_test = ood_scores_test.reshape(-1,1)

# ----- Train OoD detector using validation data -----
from lib.metric import get_metrics, train_lr
lr = train_lr(ind_features_val_for_train, ood_features_val_for_train)

# ----- Calculating metrics using test data -----
metrics = get_metrics(lr, ind_features_test, ood_features_test, acc_type="best")
print("metrics: ", metrics)