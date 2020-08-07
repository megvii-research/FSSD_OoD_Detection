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
    get_img_size,
    get_inp_channel,
)
from lib.utils import split_dataloader
import argparse


parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-i','--ind', type=str, help='in distribution dataset', required=True)
parser.add_argument('-o','--ood', type=str, help='out of distribution dataset', required=True)
parser.add_argument('-m','--model_arch', type=str, help='model architecture', required=True)
parser.add_argument('--dataroot',type=str, help='datatset stroage directory',default='/data/datasets')
parser.add_argument('--batch_size',type=int,default=512)
parser.add_argument('--inp_process', action='store_true', help='whether do input pre-processing')

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
input_process = args['inp_process']

ind_test_loader = get_dataloader(args['ind'], transform, "test", dataroot=args['dataroot'], batch_size=batch_size)
ood_test_loader = get_dataloader(args['ood'], transform, "test", dataroot=args['dataroot'], batch_size=batch_size)
ind_dataloader_val_for_train, ind_dataloader_val_for_test, ind_dataloader_test = split_dataloader(args['ind'], ind_test_loader, [500, 500, -1], random=True)
ood_dataloader_val_for_train, ood_dataloader_val_for_test, ood_dataloader_test = split_dataloader(args['ood'], ood_test_loader, [500,500, -1], random=True)


from lib.inference import get_feature_dim_list
from lib.inference.FSS import (
        compute_fss,
        get_FSS_score_ensem,
        get_FSS_score_ensem_process,
        search_FSS_hyperparams
    )
from lib.metric import get_metrics, train_lr

# ----- Calcualte FSS -----
feature_dim_list,_ = get_feature_dim_list(model, img_size, inp_channel, flat=True)
print(feature_dim_list)
if args['ind'] != 'genomics': 
    # Calulate FSS using noise for image datasets
    fss = compute_fss(model, len(feature_dim_list), img_size, inp_channel, )
else:  
    # Calculate FSS using the center of validation data features for genomics dataset
    fss = compute_fss(model, len(feature_dim_list), img_size, inp_channel, ind_dataloader_val_for_train, feature_dim_list, batch_size)
layer_indexs = list(range(len(feature_dim_list)))

# ----- Calculate best magnitude for input pre-processing -----
if input_process:
    best_magnitude = search_FSS_hyperparams(model,
                                fss,
                                layer_indexs,
                                ind_dataloader_val_for_train, 
                                ood_dataloader_val_for_train,
                                ind_dataloader_val_for_test, 
                                ood_dataloader_val_for_test, 
                                std=std)

# ----- Calculate FSSD -----
if not input_process: # when no input pre-processing is used
    print('Get FSSD for in-distribution validation data.')
    ind_features_val_for_train = get_FSS_score_ensem(model, ind_dataloader_val_for_train, fss, layer_indexs)
    print('Get FSSD for OoD validation data.')
    ood_features_val_for_train = get_FSS_score_ensem(model, ood_dataloader_val_for_train, fss, layer_indexs)


    print('Get FSSD for in-distribution test data.')
    ind_features_test = get_FSS_score_ensem(model, ind_dataloader_test, fss, layer_indexs)
    print('Get FSSD for OoD test data.')
    ood_features_test = get_FSS_score_ensem(model, ood_dataloader_test, fss, layer_indexs)[:len(ind_features_test)]
else: # when input pre-processing is used
    print('Get FSSD for in-distribution validation data.')
    ind_features_val_for_train = get_FSS_score_ensem_process(model, ind_dataloader_val_for_train, fss, layer_indexs, best_magnitude, std)
    print('Get FSSD for OoD validation data.')
    ood_features_val_for_train = get_FSS_score_ensem_process(model, ood_dataloader_val_for_train, fss, layer_indexs, best_magnitude, std)


    print('Get FSSD for in-distribution test data.')
    ind_features_test = get_FSS_score_ensem_process(model, ind_dataloader_test, fss, layer_indexs, best_magnitude, std)
    print('Get FSSD for OoD test data.')
    ood_features_test = get_FSS_score_ensem_process(model, ood_dataloader_test, fss, layer_indexs, best_magnitude, std)[:len(ind_features_test)]

# ----- Training OoD detector using validation data -----
lr = train_lr(ind_features_val_for_train, ood_features_val_for_train)


# ----- Calculating metrics using test data -----
logger.info("ind_features_test shape: {}".format(ind_features_test.shape))
logger.info("ood_features_test shape: {}".format(ood_features_test.shape))

metrics = get_metrics(lr, ind_features_test, ood_features_test, acc_type="best")
print("metrics:", metrics)
