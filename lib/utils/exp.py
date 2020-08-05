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
from lib.model.facerec import resnet as facerec_resnet
from lib.model.facerec import resnet_wo_head as facerec_resnet_wo_head
from lib.model.rnn import BiRNN
from lib.model import resnet_imagenet
from lib.dataLoader import getTargetDataSet # TODO: test all dataset results
# from lib.dataLoader import get_dataset # Dogs dataset debug
# from lib.dataLoader.imagenet_shift import (
#     getImageNetTestDataSet,
#     getImageNetTrainDataSet,
# )
# from lib.dataLoader.celeba import getCelebADataSet, getCelebABlurDataSet
# from lib.dataLoader.genomics import getGenomicsDataSet
from lib.inference import MCDropout
import random


key2model_path = {
    
    # "ms1m_resnext_oe"       : "s3://fss-research/trained_models/OE_ms1m_train_amsmx.pth",
    # "celeba_resnext_oe"     : "s3://fss-research/trained_models/OE_celeba_resnext_close_set.pth",
    # "dogs50B_resnet34_oe"   : "s3://fss-research/trained_models/OE_dogs50A_resnet34.pth",
    "fmnist_lenet": "pre_trained/lenet_fmnist.pth",
    "cifar10_resnet": "pre_trained/resnet_cifar10.pth",
    "fmnist_lenet_oe"       : "pre_trained/lenet_fmnist_oe.pth",
    "cifar10_resnet_oe"     : "pre_trained/resnet_cifar10_oe.pth",
    # "ms1m_resnext": "s3://fss-research/trained_models/ms1m_train_amsmx.pth",
    # "ms1m_resnext_triplet": "/data/jupyter/data_quality_ablation_study/exp_data/triplet_relu/model.pth.step_175000",
    # # "celeba_resnext": "s3://fss-research/trained_models/celeba_resnext.pth",
    # "celeba_resnext": "s3://fss-research/trained_models/celeba_resnext_close_set.pth",
    # "imagenet_resnet": "/data/datasets/resnet50_imagenet.pth", # download from https://download.pytorch.org/models/resnet50-19c8e357.pth
    # "dogs50A_resnet": "s3://fss-research/trained_models/dogs50A-resnet34-ensemble-css/dogs50A_resnet34-1.pth",
    # "dogs50B_resnet": "s3://fss-research/trained_models/dogs50A-resnet34-ensemble-css/dogs50A_resnet34-1.pth",
    # "dogs100_resnet": "s3://fss-research/trained_models/dogs100_resnet34_ensemble_css/dogs100_resnet34-1.pth",
    # "genomics_ind_rnn": "s3://fss-research/trained_models/genomics_rnn.pth",
}


def get_dropout_rate(dataset):
    dropout_rate_dict = {
        "cifar10": 0.05,
        "fmnist": 0.5,  # use mcd var metric, auroc=81.6, checked. 0601
        "celeba": 0.2,
        "ms1m": 0.2, # use mcd mu metric, auroc=67.2, checked. 0604
        "dogs50A": 0.4, # use mcd mu metric, auroc=67.2, checked. 0604
    }
    return dropout_rate_dict.get(dataset, 0.2)


key2model_arch = {
    "cifar10_resnet": resnet.ResNet34(num_c=10),
    "fmnist_lenet": lenet.LeNet(),
    "ms1m_resnext": facerec_resnet.ResNeXt(),
    "ms1m_resnext_triplet": facerec_resnet_wo_head.ResNeXt(),
    "imagenet_resnet": resnet_imagenet.resnet50(),
    "celeba_resnext": facerec_resnet.ResNeXt(num_classes=10122),
    "dogs50A_resnet": resnet34(num_classes=50),
    "dogs50B_resnet": resnet34(num_classes=50),
    "genomics_ind_rnn": BiRNN(hidden_dim=1000, num_layers=2),
    "cifar10_mcd_resnet": MCDropout.get_model("cifar10", get_dropout_rate("cifar10")),
    "fmnist_mcd_lenet": MCDropout.get_model("fmnist", get_dropout_rate("fmnist")),
    # "ms1m_mcd_resnext": MCDropout.get_model("ms1m", get_dropout_rate("ms1m")),
    "celeba_mcd_resnext": MCDropout.get_model("celeba", get_dropout_rate("celeba")),
    "dogs50A_mcd_resnext": MCDropout.get_model("dogs50A", get_dropout_rate("dogs50A")),
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

    if "mcd" in model_arch:
        try:
            if type(model) is dict:
                MCDropout.load_resnext_model(
                    model["backbone"], model["head"], pretrained_model_path=model_path
                )
            else:
                MCDropout.load_model(model, model_path)
            return model
        except:
            print("something went wrong while loading: {}".format(model_path))
            raise NotImplementedError

    logger.info("load state dict")
    # weight = load_pth(model_path)
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
    if 'cifar' in key:
        return (0.4914, 0.4822, 0.4465)
    if 'fmnist' in key:
        return (0.2860,)
    if 'dogs100' in key:
        return (0.485, 0.456, 0.406)
    return {
        "cifar10": (0.4914, 0.4822, 0.4465),
        "fmnist": (0.2860,),
        "ms1m": (0.5, 0.5, 0.5),
        "celeba": (0.5, 0.5, 0.5),
        "imagenet": (0.485, 0.456, 0.406),
        "dogs50A": (0.485, 0.456, 0.406),
        "dogs50B": (0.485, 0.456, 0.406),
        "dogs100": (0.485, 0.456, 0.406),
        "dogs100_resize": (0.485, 0.456, 0.406),
        "svhn": (0.4376821046090723, 0.4437697045639686, 0.4728044222297267),
        "mnist": (0.1307, ),
        "emnist": (0.1751, ),
        "tiny_imagenet": (0.47702616085763766, 0.4803492033918661, 0.4803173768346797),
        "genomics_ind": None,
        # TODO
    }[key]

def get_std(ind):
    key = ind
    if 'cifar' in key:
        return  (0.2023, 0.1994, 0.2010)
    if 'fmnist' in key:
        return (0.3530,)
    if 'dogs100' in key:
        return (0.229, 0.224, 0.225)
    return {
        "cifar10": (0.2023, 0.1994, 0.2010),
        "fmnist": (0.3530,),
        "ms1m": (1.0, 1.0, 1.0),
        "celeba": (1.0, 1.0, 1.0),
        "imagenet": (0.229, 0.224, 0.225),
        "dogs50A": (0.229, 0.224, 0.225),
        "dogs50B": (0.229, 0.224, 0.225),
        "dogs100": (0.229, 0.224, 0.225),
        "dogs100_resize": (0.229, 0.224, 0.225),
        "svhn": (0.19803012447157134, 0.20101562471828877, 0.19703614172172396),
        "mnist": (0.3081, ),
        "emnist": (0.3332, ),
        "tiny_imagenet": (0.30890223096246455, 0.30437906595277503, 0.3016761812281734),
        "genomics_ind": None,
        # TODO
    }[key]


def get_raw_transform(ind): 
    """
    transformation without normalization
    for image corruption experiments
    """
    key = ind
    if 'cifar10' in key:
        return transforms.Compose([transforms.Resize(get_img_size(key)), transforms.ToTensor()])
    if 'fmnist' in key:
        return transforms.Compose([transforms.ToTensor() ])
    if 'dogs100' in key:
        return transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(get_img_size(key)), transforms.ToTensor() ])
    return {
        "cifar10": transforms.Compose([transforms.Resize(get_img_size(key)), transforms.ToTensor()]),
        "tiny_imagenet": transforms.Compose([transforms.Resize(get_img_size(key)), transforms.ToTensor() ]),
        "svhn": transforms.Compose([transforms.Resize(get_img_size(key)), transforms.ToTensor()]),
        "fmnist": transforms.Compose([transforms.ToTensor()]),
        "mnist": transforms.Compose([transforms.ToTensor() ]),
        "emnist": transforms.Compose([transforms.ToTensor()]),
        "ms1m": transforms.Compose([transforms.Resize(get_img_size(key)), transforms.ToTensor() ]),
        "celeba": transforms.Compose([transforms.Resize(get_img_size(key)), transforms.ToTensor() ]),
        "imagenet": transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(get_img_size(key)), transforms.ToTensor()]),
        "dogs50A": transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(get_img_size(key)), transforms.ToTensor()]),
        "dogs50B": transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(get_img_size(key)), transforms.ToTensor()]),
        "dogs100": transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(get_img_size(key)), transforms.ToTensor(),
             ]),
        # TODO
        "dogs100_resize": transforms.Compose(
            [transforms.Resize(get_img_size(key)), transforms.CenterCrop(get_img_size(key)), transforms.ToTensor(),
             ]),
        "genomics_ind": None,
        # "dogs100_resize": transforms.Compose([transforms.Resize(get_img_size(key)), transforms.CenterCrop(get_img_size(key)), transforms.ToTensor(), transforms.Normalize(get_mean("celeba"), get_std("celeba")),])
    }[key]


def get_normalize_transform(ind):
    """
    normalization transformation
    for image corruption experiments
    """
    key = ind
    if 'cifar10' in key:
        return transforms.Compose([transforms.Normalize(get_mean(key), get_std(key)), ])
    if 'fmnist' in key:
        return transforms.Compose([transforms.Normalize(get_mean(key), get_std(key)), ])
    if 'dogs100' in key:
        return transforms.Compose(
            [transforms.Normalize(get_mean(key), get_std(key)), ])
    return {
        "cifar10": transforms.Compose([transforms.Normalize(get_mean(key), get_std(key)), ]),
        "tiny_imagenet": transforms.Compose([transforms.Normalize(get_mean(key), get_std(key)), ]),
        "svhn": transforms.Compose([transforms.Normalize(get_mean(key), get_std(key)), ]),
        "fmnist": transforms.Compose([transforms.Normalize(get_mean(key), get_std(key)), ]),
        "mnist": transforms.Compose([transforms.Normalize(get_mean(key), get_std(key)), ]),
        "emnist": transforms.Compose([transforms.Normalize(get_mean(key), get_std(key)), ]),
        "ms1m": transforms.Compose([transforms.Normalize(get_mean(key), get_std(key)), ]),
        "celeba": transforms.Compose([transforms.Normalize(get_mean(key), get_std(key)), ]),
        "imagenet": transforms.Compose(
            [transforms.Normalize(get_mean(key), get_std(key)), ]),
        "dogs50A": transforms.Compose(
            [transforms.Normalize(get_mean(key), get_std(key)), ]),
        "dogs50B": transforms.Compose(
            [transforms.Normalize(get_mean(key), get_std(key)), ]),
        "dogs100": transforms.Compose(
            [transforms.Normalize(get_mean(key), get_std(key)), ]),
        # TODO
        "dogs100_resize": transforms.Compose(
            [transforms.Normalize(get_mean(key), get_std(key)), ]),
        "genomics_ind": None,
        # "dogs100_resize": transforms.Compose([transforms.Resize(get_img_size(key)), transforms.CenterCrop(get_img_size(key)), transforms.ToTensor(), transforms.Normalize(get_mean("celeba"), get_std("celeba")),])
    }[key]


def get_transform(ind):
    key = ind
    return {
        "cifar10": transforms.Compose([transforms.Resize(get_img_size(key)), transforms.ToTensor(), transforms.Normalize(get_mean(key), get_std(key)),]),
        "tiny_imagenet": transforms.Compose([transforms.Resize(get_img_size(key)), transforms.ToTensor(), transforms.Normalize(get_mean(key), get_std(key)),]),
        "svhn": transforms.Compose([transforms.Resize(get_img_size(key)), transforms.ToTensor(), transforms.Normalize(get_mean(key), get_std(key)),]),
        "fmnist": transforms.Compose([transforms.ToTensor(), transforms.Normalize(get_mean(key), get_std(key)),]),
        "mnist": transforms.Compose([transforms.ToTensor(), transforms.Normalize(get_mean(key), get_std(key)),]),
        "emnist": transforms.Compose([transforms.ToTensor(), transforms.Normalize(get_mean(key), get_std(key)),]),
        "ms1m": transforms.Compose([transforms.Resize(get_img_size(key)), transforms.ToTensor(), transforms.Normalize(get_mean(key), get_std(key)),]),
        "celeba": transforms.Compose([transforms.Resize(get_img_size(key)), transforms.ToTensor(), transforms.Normalize(get_mean(key), get_std(key)),]),
        "imagenet": transforms.Compose([transforms.Resize(256), transforms.CenterCrop(get_img_size(key)), transforms.ToTensor(), transforms.Normalize(get_mean(key), get_std(key)),]),
        "dogs50A": transforms.Compose([transforms.Resize(256), transforms.CenterCrop(get_img_size(key)), transforms.ToTensor(), transforms.Normalize(get_mean(key), get_std(key)),]),
        "dogs50B": transforms.Compose([transforms.Resize(256), transforms.CenterCrop(get_img_size(key)), transforms.ToTensor(), transforms.Normalize(get_mean(key), get_std(key)),]),
        "dogs100": transforms.Compose([transforms.Resize(256), transforms.CenterCrop(get_img_size(key)), transforms.ToTensor(), transforms.Normalize(get_mean(key), get_std(key)),]),
        # TODO
        "dogs100_resize": transforms.Compose([transforms.Resize(get_img_size(key)), transforms.CenterCrop(get_img_size(key)), transforms.ToTensor(), transforms.Normalize(get_mean(key), get_std(key)),]),
        "genomics_ind": None,
        # "dogs100_resize": transforms.Compose([transforms.Resize(get_img_size(key)), transforms.CenterCrop(get_img_size(key)), transforms.ToTensor(), transforms.Normalize(get_mean("celeba"), get_std("celeba")),])
    }[key]

# def get_dataloader_train(dataset_name, transform, dataroot='/data/datasets', batch_size=512):
#     """
#     training dataloader
#     """
#     key = dataset_name
#     if key in ['cifar10', 'svhn', 'emnist', 'fmnist', 'mnist', 'dogs50A', 'dogs50B', 'non-dogs']:    
#         dataset = get_dataset(key, batch_size, transform, dataroot, split="train")
#         return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4)

# def get_dataloader_test(dataset_name, transform, dataroot='/data/datasets', split=[500,500, -1], batch_size=512):
#     """
#     test dataloader, dataset splitted according to "split" 
#     """
#     key = dataset_name
#     if key in ['cifar10', 'svhn', 'emnist', 'fmnist', 'mnist', 'dogs50A', 'dogs50B', 'non-dogs']:    
#         dataset = get_dataset(key, batch_size, transform, dataroot, split="test")
#         indices = list(range(len(dataset)))
#         random.shuffle(indices)
        
#         val_train_split = split[0]
#         val_test_split = split[0] + split[1]
#         val_train_indices = indices[:val_train_split]
#         val_test_indices = indices[val_train_split:val_test_split]
#         test_indices = indices[val_test_split:]

#         samplers = {
#             'val_train': torch.utils.data.SubsetRandomSampler(val_train_indices),
#             'val_test': torch.utils.data.SubsetRandomSampler(val_test_indices),
#             'test': torch.utils.data.SubsetRandomSampler(test_indices)
#         }

#         val_train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=samplers['val_train'], num_workers=4)
#         val_test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=samplers['val_test'], num_workers=4)
#         test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=samplers['test'], num_workers=4)
#         return val_train_loader, val_test_loader, test_loader



def get_dataloader(dataset_name, transform, type, dataroot='/data/datasets', batch_size=512):
    assert type in ["train", "test"]
    key = dataset_name
    # dataloader = getTargetDataSet(key, batch_size, transform, dataroot, split=type)
    if key in ['cifar10', 'svhn', 'fmnist', 'mnist', 'dogs50A', 'dogs50B', 'non-dogs']:    
        dataloader = getTargetDataSet(key, batch_size, transform, dataroot, split=type)   
    # elif key == "celeba":
    #     if type == "train":
    #         dataloader, _ = getCelebADataSet(64, transform ,dataroot)
    #     else:
    #         _, dataloader = getCelebADataSet(64, transform, dataroot)
    # elif key == "celeba_blur":
    #     assert type == "test"
    #     dataloader = getCelebABlurDataSet(64, transform, dataroot)
    # elif key == "imagenet":
    #     if type == "train":
    #         dataloader = getImageNetTrainDataSet(64, transform)
    #     else:
    #         dataloader = getImageNetTestDataSet(64, transform)

    # elif key == "genomics_ind":
    #     if type == "train":
    #         dataloader = getGenomicsDataSet(64, transform, "train", dataroot)
    #     else:
    #         dataloader = getGenomicsDataSet(64, transform, "ind_test", dataroot)
    # elif key == "genomics_ood":
    #     assert type == "test"
    #     dataloader = getGenomicsDataSet(64, transform, "ood_test", dataroot)
    else:
        print(key)
        raise NotImplementedError
    return dataloader

def get_img_size(ind):
    key = ind
    return {
        "cifar10": 32,
        "cifar10-corrupt":32,
        "cifar10-part":32,
        "fmnist": 28,
        "fmnist-part":28,
        "fmnist-corrupt":28,
        "ms1m": 112,
        "celeba": 112,
        "imagenet": 224,
        "dogs50A":224,
        "dogs50B":224,
        "dogs100":224,
        "dogs100-corrupt":224,
        "dogs100-part":224,
        # TODO
        "dogs100_resize": 112,
        "genomics_ind": -1,
        "mnist": 28,
        "emnist": 28,
        "tiny_imagenet": 32,
        "svhn": 32,
        "celeba_blur": 112,
        "non-dogs": 224,
        "IJB-C": 112,
    }[key]


def get_inp_channel(ind):
    key = ind
    return {
        "cifar10": 3,
        "cifar10-corrupt":3,
        "cifar10-part":3,
        "fmnist": 1,
        "fmnist-part":1,
        "fmnist-corrupt":1,
        "mnist": 1,
        "emnist": 1,
        "ms1m": 3,
        "celeba": 3,
        "imagenet": 3,
        "tiny_imagenet": 3,
        "svhn": 3,
        "dogs50A":3,
        "dogs50B":3,
        "dogs100":3,
        "dogs100-corrupt":3,
        "dogs100-part":3,
        # TODO
        "dogs100_resize": 3,
        "genomics_ind": -1,
        "celeba_blur": 3,
        "non-dogs": 3,
        "IJB-C": 3,
    }[key]
