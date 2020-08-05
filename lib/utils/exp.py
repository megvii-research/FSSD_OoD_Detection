import os
import sys
import types
import pickle
from copy import deepcopy

from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from loguru import logger
import torchvision.models as models
from lib.model.resnet_imagenet import resnet34
from lib.model import resnet, lenet
from lib.model import resnet_imagenet
from lib.dataLoader import getTargetDataSet 
import random


key2model_path = {
    "fmnist_lenet": "pre_trained/lenet_fmnist.pth",
    "cifar10_resnet": "pre_trained/resnet_cifar10.pth",
    "fmnist_lenet_oe"       : "pre_trained/lenet_fmnist_oe.pth",
    "cifar10_resnet_oe"     : "pre_trained/resnet_cifar10_oe.pth",
}



key2model_arch = {
    "cifar10_resnet": resnet.ResNet34(num_c=10),
    "fmnist_lenet": lenet.LeNet(),
}

def get_modeldir_ens(ind, model_arch):
    return f'./pre_trained/{model_arch}_{ind}_ensemble/'


def get_model(ind, model_arch, test_oe=False, target_model_path=None):
    key = ind + "_" + model_arch
    model = key2model_arch[key]
    if test_oe:
        key += "_oe"
    if target_model_path is None:
        model_path = key2model_path[key]
    else:
        model_path = target_model_path

    logger.info("load state dict")
    weight = torch.load(model_path)
    if type(weight) == dict and "net" in weight.keys():
        weight = weight['net']
    try:
        model.load_state_dict(weight)
    except:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(weight)
    model = model.cuda()
    model.eval()
    return model

def get_mean(ind):
    key = ind
    return {
        "cifar10": (0.4914, 0.4822, 0.4465),
        "fmnist": (0.2860,),
        "svhn": (0.4376821046090723, 0.4437697045639686, 0.4728044222297267),
        "mnist": (0.1307, ),
    }[key]

def get_std(ind):
    key = ind
    return {
        "cifar10": (0.2023, 0.1994, 0.2010),
        "fmnist": (0.3530,),
        "mnist": (0.3081,),
        "svhn": (0.19803012447157134, 0.20101562471828877, 0.19703614172172396),
    }[key]


def get_raw_transform(ind): 
    """
    transformation without normalization
    for image corruption experiments
    """
    key = ind
    return {
        "cifar10": transforms.Compose([transforms.Resize(get_img_size(key)), transforms.ToTensor()]),
        "svhn": transforms.Compose([transforms.Resize(get_img_size(key)), transforms.ToTensor()]),
        "fmnist": transforms.Compose([transforms.ToTensor()]),
        "mnist": transforms.Compose([transforms.ToTensor() ]),
    }[key]


def get_normalize_transform(ind):
    """
    normalization transformation
    for image corruption experiments
    """
    key = ind
    return {
        "cifar10": transforms.Compose([transforms.Normalize(get_mean(key), get_std(key)), ]),
        "svhn": transforms.Compose([transforms.Normalize(get_mean(key), get_std(key)), ]),
        "fmnist": transforms.Compose([transforms.Normalize(get_mean(key), get_std(key)), ]),
        "mnist": transforms.Compose([transforms.Normalize(get_mean(key), get_std(key)), ]),
    }[key]


def get_transform(ind):
    key = ind
    return {
        "cifar10": transforms.Compose([transforms.Resize(get_img_size(key)), transforms.ToTensor(), transforms.Normalize(get_mean(key), get_std(key)),]),
        "svhn": transforms.Compose([transforms.Resize(get_img_size(key)), transforms.ToTensor(), transforms.Normalize(get_mean(key), get_std(key)),]),
        "fmnist": transforms.Compose([transforms.ToTensor(), transforms.Normalize(get_mean(key), get_std(key)),]),
        "mnist": transforms.Compose([transforms.ToTensor(), transforms.Normalize(get_mean(key), get_std(key)),]),
    }[key]


def get_dataloader(dataset_name, transform, type, dataroot='/data/datasets', batch_size=512):
    assert type in ["train", "test"]
    key = dataset_name
    if key in ['cifar10', 'svhn', 'fmnist', 'mnist']:    
        dataloader = getTargetDataSet(key, batch_size, transform, dataroot, split=type)   
    else:
        print(key)
        raise NotImplementedError
    return dataloader

def get_img_size(ind):
    key = ind
    return {
        "cifar10": 32,
        "fmnist": 28,
        "mnist": 28,
        "svhn": 32,
    }[key]


def get_inp_channel(ind):
    key = ind
    return {
        "cifar10": 3,
        "fmnist": 1,
        "mnist": 1,
        "svhn": 3,
    }[key]
