import os
import sys
import math
import torch
import pickle
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.models as models
from loguru import logger
from tqdm import tqdm
from torchvision import transforms

from lib.utils.exp import (
    get_model,
    get_modeldir_ens,
    get_transform,
    get_mean, 
    get_std,
    get_dataloader,
)
from lib.model import resnet,lenet
from lib.utils import split_dataloader


parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-i','--ind', type=str, help='in distribution dataset', required=True)
parser.add_argument('-o','--ood', type=str, help='out of distribution dataset', required=True)
parser.add_argument('-m','--model_arch', type=str, help='model architecture', required=True)
parser.add_argument('-b','--batch_size', type=int, default=64)
parser.add_argument('--model_num',type=int, help='the number of classifiers for ensemble',default=5)
parser.add_argument('--dataroot',type=str, help='datatset stroage directory',default='/data/datasets')
args = vars(parser.parse_args())
print(args)

modeldir = get_modeldir_ens(args['ind'], args['model_arch'])
ensemble_num = args['model_num']


# ----- load dataset -----
transform = get_transform(args['ind'])
std = get_std(args['ind'])

ind_test_loader = get_dataloader(args['ind'], transform, "test",dataroot=args['dataroot'],batch_size=args['batch_size'])
ood_test_loader = get_dataloader(args['ood'], transform, "test",dataroot=args['dataroot'],batch_size=args['batch_size'])
ind_dataloader_val_for_train, ind_dataloader_val_for_test, ind_dataloader_test = split_dataloader(args['ind'], ind_test_loader, [500,500,-1])
ood_dataloader_val_for_train, ood_dataloader_val_for_test, ood_dataloader_test = split_dataloader(args['ood'], ood_test_loader, [500,500,-1])

# ----- Calculating and averaging maximum softmax probabilities -----
from lib.inference.ODIN import get_ODIN_score
best_temperature = 1.0
best_magnitude = 0.0

ind_ensemble_val = []
ood_ensemble_val = []
ind_ensemble_test = []
ood_ensemble_test = []
for id, ckpt in enumerate(os.listdir(modeldir)[:ensemble_num]):
    model_path = modeldir + args['ind'] + '_' + args['model_arch'] + f'_{id}.pth'
    model = get_model(args['ind'], args['model_arch'], target_model_path=model_path)

    ind_scores_val_for_train = get_ODIN_score(model, ind_dataloader_val_for_train, best_magnitude, best_temperature, std=std)
    ood_scores_val_for_train = get_ODIN_score(model, ood_dataloader_val_for_train, best_magnitude, best_temperature, std=std)
    ind_ensemble_val.append(ind_scores_val_for_train)
    ood_ensemble_val.append(ood_scores_val_for_train)

    ind_scores_test = get_ODIN_score(model, ind_dataloader_test, best_magnitude, best_temperature, std=std)
    ood_scores_test = get_ODIN_score(model, ood_dataloader_test, best_magnitude, best_temperature, std=std)
    ind_ensemble_test.append(ind_scores_test)
    ood_ensemble_test.append(ood_scores_test)

take_mean_and_reshape = lambda x: np.array(x).mean(axis=0).reshape(-1, 1)
ind_val, ood_val, ind_test, ood_test = map(take_mean_and_reshape, [ind_ensemble_val, ood_ensemble_val, ind_ensemble_test, ood_ensemble_test])

# ----- Train OoD detector using validation data -----
from lib.metric import get_metrics, train_lr
lr = train_lr(ind_val, ood_val)

# ----- Calculating metrics using test data -----
metrics = get_metrics(lr, ind_test, ood_test, acc_type="best")
print("metrics: ", metrics)

