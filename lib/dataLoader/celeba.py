import os
import sys

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.utils.data as data
import numpy as np

from lib.utils.io import load_pkl
from lib.dataLoader.common import InferenceDataLoader

def getCelebADataSet(batch_size, transform, dataroot):
    dataset_dir = os.path.join(dataroot, "celeba")
    if not os.path.exists(dataset_dir):
        print(r"""
            dataset "/data/dataset/celeba" does not exist
            download dataset from s3://fss-research/dataset/celeba
        """)
        raise FileNotFoundError(dataset_dir)
    train_dir = os.path.join(dataset_dir, "new_train")

    train_nori = load_pkl(os.path.join(train_dir, "align5p.nori_id"))
    train_info = load_pkl(os.path.join(train_dir, "info"))
    # labels = np.zeros(len(train_nori), dtype='int64')
    labels = []
    sel_noris = []
    for i,(s,e) in enumerate(train_info):
        e = min(e,s+5)
        sel_noris.extend(train_nori[s:e])
        labels.extend([i]*(e-s))
    train_loader = InferenceDataLoader(sel_noris, 112, transform, labels=labels, shuffle=True, batch_size=batch_size)

    test_dir = os.path.join(dataset_dir, "new_test")
    test_nori = load_pkl(os.path.join(test_dir, "align5p.nori_id"))
    test_nori = np.array(test_nori)
    np.random.seed(42)
    random_idx = np.random.permutation(list(range(len(test_nori))))[:50000]
    test_loader = InferenceDataLoader(test_nori[random_idx], 112, transform, batch_size=batch_size)
    # test_loader = InferenceDataLoader(test_nori, -1, transform, batch_size=batch_size)
    return train_loader, test_loader

def getCelebABlurDataSet(batch_size, transform, dataroot):
    dataset_dir = os.path.join(dataroot, "celeba/blur")
    if not os.path.exists(dataset_dir):
        print(r"""
            dataset "/data/datasets/IJB_C" does not exist
            download dataset from s3://fss-research/dataset/IJB-C
        """)
        raise FileNotFoundError(dataset_dir)

    align5p_nori_id = load_pkl(os.path.join(dataset_dir, "align5p.nori_id"))
    align5p_nori_id = np.array(align5p_nori_id)
    np.random.seed(42)
    random_idx = np.random.permutation(list(range(len(align5p_nori_id))))[:50000]
    test_loader = InferenceDataLoader(align5p_nori_id[random_idx], 112, transform, batch_size=batch_size)
    # test_loader = InferenceDataLoader(align5p_nori_id, -1, transform, batch_size=batch_size)
    return test_loader
