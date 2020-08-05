import os
import sys
import pickle
from copy import deepcopy

import numpy as np
import torch

def split_dataloader(datasetname, dataloader, sizes=[1000,-1], random=False, seed=42):
    assert datasetname in ["cifar10", "cifar100", "svhn", "cifar10_shift", "fmnist", "mnist", 
                           "tiny_imagenet", "tiny_imagenet_crop", "tiny_imagenet_resize","dogs50B_shift",
                           "ms1m", "IJB-C", "imagenet", "imagenet_shift", "noise", "celeba", 
                           "celeba_blur", "dogs50A","dogs50B","non-dogs","dogs100","cifar10-corrupt",
                           "cifar10-part","fmnist-corrupt","fmnist-part","dogs100-corrupt","dogs100-part",
                           "genomics_ind", "genomics_ood"], datasetname

    if datasetname in ["cifar10_shift", "ms1m", "IJB-C", "imagenet", "imagenet_shift", "celeba", "celeba_blur","dogs50B_shift"]:
        return split_dataloader_by_nori(dataloader, sizes, random, seed)
    if datasetname in ["tiny_imagenet", "tiny_imagenet_crop", "tiny_imagenet_resize"]:
        return split_dataloader_tiny_imagenet(dataloader, sizes, random, seed)
    if datasetname in ["dogs50A","dogs50B","non-dogs","dogs100","dogs100-part","dogs100-corrupt"]:
        dataloaders = split_dataloader_dogs(dataloader, sizes, random, seed)
        return dataloaders
    if datasetname in ["genomics_ind", "genomics_ood"]:
        return split_dataloader_genomics(dataloader, sizes, random, seed)
        
    dataloaders = []
    data = dataloader.dataset.data
    if "targets" in dataloader.dataset.__dict__.keys():
        targets = np.array(dataloader.dataset.targets)
    elif "labels" in dataloader.dataset.__dict__.keys():
        targets = np.array(dataloader.dataset.labels)
    else:
        targets = None
    
    total = len(dataloader.dataset)
    if random:
        np.random.seed(seed)
        idxs = np.random.permutation(list(range(total)))
    else:
        idxs = list(range(total))
    s = 0
    for size in sizes:
        if size == -1:
            t = deepcopy(dataloader)
            t.dataset.data = data[idxs[s:]]
            t.sampler.data_source.data = data[idxs[s:]]
            if datasetname in ["cifar10", "cifar100", "fmnist", "mnist","cifar10-part","fmnist-part"]:
                t.targets = targets[idxs[s:]].tolist()
                t.sampler.data_source.targets = targets[idxs[s:]].tolist()
            elif datasetname in ["svhn"]:
                t.labels = targets[idxs[s:]].tolist()
                t.sampler.data_source.labels = targets[idxs[s:]].tolist()
            elif datasetname in ['noise']:
                pass
            else:
                raise NotImplementedError
        else:
            t = deepcopy(dataloader)
            t.dataset.data = data[idxs[s:s+size]]
            t.sampler.data_source.data = data[idxs[s:s+size]]
            if datasetname in ["cifar10", "cifar100", "fmnist", "mnist","cifar10-corrupt","cifar10-part","fmnist-part","fmnist-corrupt"]:
                t.targets = targets[idxs[s:s+size]].tolist()
                t.sampler.data_source.targets = targets[idxs[s:s+size]].tolist()
            elif datasetname in ["svhn"]:
                t.labels = targets[idxs[s:s+size]].tolist()
                t.sampler.data_source.labels = targets[idxs[s:s+size]].tolist()
            elif datasetname in ['noise']:
                pass
            else:
                raise NotImplementedError
            s += size
        dataloaders.append(t)
    return dataloaders


def split_dataloader_by_nori(dataloader, sizes=[1000,-1], random=False, seed=42):
    align5p_nori_id = dataloader.dataset.align5p_nori_id
    align5p_nori_id = np.array(align5p_nori_id)
    total = len(dataloader.dataset)
    dataloaders = []
    if random:
        np.random.seed(seed)
        idxs = np.random.permutation(list(range(total)))
    else:
        idxs = list(range(total))
    s = 0
    for size in sizes:
        if size == -1:
            t = deepcopy(dataloader)
            t.dataset.align5p_nori_id = align5p_nori_id[idxs[s:]]
            t.sampler.data_source.align5p_nori_id = align5p_nori_id[idxs[s:]]
        else:
            t = deepcopy(dataloader)
            t.dataset.align5p_nori_id = align5p_nori_id[idxs[s:s+size]]
            t.sampler.data_source.align5p_nori_id = align5p_nori_id[idxs[s:s+size]]
            s += size
        dataloaders.append(t)
    return dataloaders


def split_dataloader_tiny_imagenet(dataloader, sizes, random, seed):
    data = dataloader.dataset.imgs
    data = np.array(data)
    targets = dataloader.dataset.targets
    targets = np.array(targets)
    total = len(dataloader.dataset)
    dataloaders = []
    if random:
        np.random.seed(seed)
        idxs = np.random.permutation(list(range(total)))
    else:
        idxs = list(range(total))
    s = 0
    for size in sizes:
        if size == -1:
            t = deepcopy(dataloader)
            t.dataset.imgs = data[idxs[s:]]
            t.dataset.samples = data[idxs[s:]]
            t.dataset.targets = targets[idxs[s:]]
            t.sampler.data_source.imgs = data[idxs[s:]]
            t.sampler.data_source.samples = data[idxs[s:]]
            t.sampler.data_source.targets = targets[idxs[s:]]
        else:
            t = deepcopy(dataloader)
            t.dataset.imgs = data[idxs[s:s+size]]
            t.dataset.samples = data[idxs[s:s+size]]
            t.dataset.targets = targets[idxs[s:s+size]]
            t.sampler.data_source.imgs = data[idxs[s:s+size]]
            t.sampler.data_source.samples = data[idxs[s:s+size]]
            t.sampler.data_source.targets = targets[idxs[s:s+size]]
            s += size
        dataloaders.append(t)
    return dataloaders

def split_dataloader_dogs(dataloader, sizes, random, seed):
    imgs = dataloader.dataset.imgs
    imgs = np.array(imgs)
    data = np.array([img[0] for img in imgs]) # path
    targets = np.array([img[1] for img in imgs]) # label
    total = len(dataloader.dataset)
    dataloaders = []
    if random:
        np.random.seed(seed)
        idxs = np.random.permutation(list(range(total)))
    else:
        idxs = list(range(total))
    s = 0
    for size in sizes:
        if size == -1:
            t = deepcopy(dataloader)
            t.dataset.imgs = data[idxs[s:]]
            t.dataset.samples = imgs[idxs[s:]]
            t.dataset.targets = targets[idxs[s:]]
            t.sampler.data_source.imgs = data[idxs[s:]]
            t.sampler.data_source.samples = imgs[idxs[s:]]
            t.sampler.data_source.targets = targets[idxs[s:]]
        else:
            t = deepcopy(dataloader)
            t.dataset.imgs = data[idxs[s:s+size]]
            t.dataset.samples = imgs[idxs[s:s+size]]
            t.dataset.targets = targets[idxs[s:s+size]]
            t.sampler.data_source.imgs = data[idxs[s:s+size]]
            t.sampler.data_source.samples = imgs[idxs[s:s+size]]
            t.sampler.data_source.targets = targets[idxs[s:s+size]]
            s += size
        dataloaders.append(t)
    return dataloaders

def split_dataloader_genomics(dataloader, sizes, random, seed):
    tensors = dataloader.dataset.tensors[0]
    labels = dataloader.dataset.tensors[1]
    total = len(dataloader.dataset)
    dataloaders = []
    if random:
        np.random.seed(seed)
        idxs = np.random.permutation(list(range(total)))
    else:
        idxs = list(range(total))
    s = 0
    for size in sizes:
        if size == -1:
            t = deepcopy(dataloader)
            t.dataset.tensors = (tensors[idxs[s:]], labels[idxs[s:]])
            t.sampler.data_source.tensors = (tensors[idxs[s:]], labels[idxs[s:]])
        else:
            t = deepcopy(dataloader)
            t.dataset.tensors = (tensors[idxs[s:s+size]], labels[idxs[s:s+size]])
            t.sampler.data_source.tensors = (tensors[idxs[s:s+size]], labels[idxs[s:s+size]])
            s += size
        dataloaders.append(t)
    return dataloaders
