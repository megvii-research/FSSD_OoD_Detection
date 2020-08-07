import os
import sys
import pickle
from copy import deepcopy

import numpy as np
import torch

def split_dataloader(datasetname, dataloader, sizes=[1000,-1], random=False, seed=42):
    assert datasetname in ["cifar10", "svhn", "fmnist", "mnist", 
                           "genomics", "genomics_ood"], datasetname
    if datasetname in ["genomics", "genomics_ood"]:
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
            if datasetname in ["cifar10", "cifar100", "fmnist", "mnist"]:
                t.targets = targets[idxs[s:]].tolist()
                t.sampler.data_source.targets = targets[idxs[s:]].tolist()
            elif datasetname in ["svhn"]:
                t.labels = targets[idxs[s:]].tolist()
                t.sampler.data_source.labels = targets[idxs[s:]].tolist()
            else:
                raise NotImplementedError
        else:
            t = deepcopy(dataloader)
            t.dataset.data = data[idxs[s:s+size]]
            t.sampler.data_source.data = data[idxs[s:s+size]]
            if datasetname in ["cifar10", "fmnist", "mnist"]:
                t.targets = targets[idxs[s:s+size]].tolist()
                t.sampler.data_source.targets = targets[idxs[s:s+size]].tolist()
            elif datasetname in ["svhn"]:
                t.labels = targets[idxs[s:s+size]].tolist()
                t.sampler.data_source.labels = targets[idxs[s:s+size]].tolist()
            else:
                raise NotImplementedError
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
