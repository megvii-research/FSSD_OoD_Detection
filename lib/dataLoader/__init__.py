# from lib.dataLoader.cifar_svhn import *

import os
import sys
import torch
import torchvision
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def getTargetDataSet(data_type, batch_size, input_TF, dataroot, split='train'):
    if data_type == "cifar10":
        data_loader = getCIFAR10(
            batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1, train=split=='train', val=split=='test'
        )
        return data_loader
    elif data_type == "svhn":
        data_loader = getSVHN(
            batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1, train=split=='train', val=split=='test'
        )
    elif data_type == "fmnist":
        data_loader = getFMNIST(
            batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1, train=split=='train', val=split=='test'
        )
    elif data_type == "mnist":
        data_loader = getMNIST(
            batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1, train=split=='train', val=split=='test'
        )
    elif data_type == "dogs50A":
        data_loader = getDogs50A(
            batch_size=batch_size, TF=input_TF, dataroot=dataroot, num_workers=20, train=split=='train', val=split=='test'
        )
    elif data_type == "dogs50B":
        data_loader = getDogs50B(
            batch_size=batch_size, TF=input_TF, dataroot=dataroot, num_workers=20, train=split=='train', val=split=='test'
        )
    elif data_type == "non-dogs":
        data_loader = getNonDogsImageNet(
            batch_size=batch_size, TF=input_TF, dataroot=dataroot, num_workers=1, train=split=='train', val=split=='test'
        )
    else:
        raise NotImplementedError

    return data_loader

def get_dataset(dataset_name, batch_size, transform, dataroot, split, **kwargs):
    if split == "test":
        data_root = os.path.expanduser(os.path.join(dataroot, f"{dataset_name}-data"))
        if dataset_name == "cifar10":
            dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
        elif dataset_name == "svhn":
            dataset = datasets.SVHN(root=data_root, split="test", download=True, transform=transform)
        elif dataset_name == "fmnist":
            dataset = datasets.FashionMNIST(root=data_root, train=False, download=True, transform=transform)
        elif dataset_name == "mnist":
            dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
        elif dataset_name == "emnist":
            dataset = datasets.EMNIST(root=data_root, split="balanced", download=True, transform=transform)
        elif dataset_name == "dogs50A" or "dogs50B":
            folder = dataroot + r"/dogs50B-val"
            dataset = torchvision.datasets.ImageFolder(folder, transform=transform)
        elif dataset_name == "non-dogs":
            folder = dataroot + r"/non-dogs-val"
            dataset = torchvision.datasets.ImageFolder(folder, transform=transform)
        else:
            print("this test set has not been implemented yet.")
            raise NotImplementedError
    
    if split == "train":
        data_root = os.path.expanduser(os.path.join(dataroot, f"{dataset_name}-data"))
        if dataset_name == "cifar10":
            dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
        elif dataset_name == "svhn":
            dataset = datasets.SVHN(root=data_root, split="train", download=True, transform=transform)
        elif dataset_name == "fmnist":
            dataset = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=transform)
        elif dataset_name == "mnist":
            dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
        elif dataset_name == "dogs50A" or "dogs50B":
            folder = dataroot + r"/dogs50A-train"
            dataset = torchvision.datasets.ImageFolder(folder, transform=transform)
        else:
            print("this training set has not been implemented yet.")
            raise NotImplementedError

    # ['cifar10', 'svhn', 'emnist', 'fmnist', 'mnist', 'dogs50A', 'dogs50B', 'non-dogs']

    return dataset


def getSVHN(
    batch_size,
    TF,
    data_root="/tmp/public_dataset/pytorch",
    train=True,
    val=True,
    **kwargs
):
    data_root = os.path.expanduser(os.path.join(data_root, "svhn-data"))
    num_workers = kwargs.setdefault("num_workers", 1)
    kwargs.pop("input_size", None)

    def target_transform(target):
        new_target = target - 1
        if new_target == -1:
            new_target = 9
        return new_target

    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.SVHN(root=data_root, split="train", download=True, transform=TF),
            batch_size=batch_size,
            shuffle=True,
            **kwargs
        )
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.SVHN(root=data_root, split="test", download=True, transform=TF),
            batch_size=batch_size,
            shuffle=False,
            **kwargs
        )
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def getCIFAR10(
    batch_size,
    TF,
    data_root="/tmp/public_dataset/pytorch",
    train=True,
    val=True,
    **kwargs
):
    data_root = os.path.expanduser(os.path.join(data_root, "cifar10-data"))
    num_workers = kwargs.setdefault("num_workers", 1)
    kwargs.pop("input_size", None)
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=data_root, train=True, download=True, transform=TF),
            batch_size=batch_size,
            shuffle=True,
            **kwargs
        )
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=data_root, train=False, download=True, transform=TF),
            batch_size=batch_size,
            shuffle=False,
            **kwargs
        )
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def getFMNIST(
    batch_size,
    TF,
    data_root="/tmp/public_dataset/pytorch",
    train=True,
    val=True,
    **kwargs
):
    data_root = os.path.expanduser(os.path.join(data_root, "fmnist-data"))
    num_workers = kwargs.setdefault("num_workers", 1)
    kwargs.pop("input_size", None)
    ds = []
    if train:
        print("Get FMNIST training data")
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(
                root=data_root, train=True, download=True, transform=TF
            ),
            batch_size=batch_size,
            shuffle=True,
            **kwargs
        )
        ds.append(train_loader)
    if val:
        print("Get FMNIST validation data")
        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(
                root=data_root, train=False, download=True, transform=TF
            ),
            batch_size=batch_size,
            shuffle=False,
            **kwargs
        )
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    
    return ds

def getMNIST(
    batch_size,
    TF,
    data_root="/tmp/public_dataset/pytorch",
    train=True,
    val=True,
    **kwargs
):
    # data_root = os.path.expanduser(os.path.join(data_root, 'mnist-data'))
    num_workers = kwargs.setdefault("num_workers", 1)
    kwargs.pop("input_size", None)
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root=data_root, train=True, download=True, transform=TF),
            batch_size=batch_size,
            shuffle=True,
            **kwargs
        )
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root=data_root, train=False, download=True, transform=TF),
            batch_size=batch_size,
            shuffle=False,
            **kwargs
        )
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def getDogs50A(batch_size, TF, dataroot, train=True, val=True, **kwargs):
    ds = []
    if train:
        folder = dataroot + r"/dogs50A-train"
        Dataset = torchvision.datasets.ImageFolder(folder, transform=TF)
        loader = torch.utils.data.DataLoader(
            Dataset, batch_size=batch_size, shuffle=True, num_workers=20
        )
        ds.append(loader)
    if val:
        folder = dataroot + r"/dogs50A-val"
        Dataset = torchvision.datasets.ImageFolder(folder, transform=TF)
        loader = torch.utils.data.DataLoader(
            Dataset, batch_size=batch_size, shuffle=True, num_workers=20
        )
        ds.append(loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def getDogs50B(batch_size, TF, dataroot, train=True, val=True, **kwargs):
    ds = []
    if train:
        folder = dataroot + r"/dogs50B-train"
        Dataset = torchvision.datasets.ImageFolder(folder, transform=TF)
        loader = torch.utils.data.DataLoader(
            Dataset, batch_size=batch_size, shuffle=True, num_workers=20
        )
        ds.append(loader)
    if val:
        folder = dataroot + r"/dogs50B-val"
        Dataset = torchvision.datasets.ImageFolder(folder, transform=TF)
        loader = torch.utils.data.DataLoader(
            Dataset, batch_size=batch_size, shuffle=True, num_workers=20
        )
        ds.append(loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def getNonDogsImageNet(batch_size, TF, dataroot, train=True, val=True, **kwargs):
    folder = dataroot + r"/non-dogs-val"
    Dataset = torchvision.datasets.ImageFolder(folder, transform=TF)
    loader = torch.utils.data.DataLoader(
        Dataset, batch_size=batch_size, shuffle=True, num_workers=20
    )
    if train:
        return [None, loader]  # only for OOD test
    else:
        return loader
