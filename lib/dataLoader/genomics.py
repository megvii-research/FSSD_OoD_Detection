import os
import sys

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.utils.data as data
import numpy as np

from lib.utils.io import load_pkl
from lib.dataLoader.common import InferenceDataLoader

def getGenomicsDataSet(batch_size, transform, type, dataroot="/data/datasets"):
    dataset_dir = os.path.join(dataroot, "llr_ood_genomics")
    if not os.path.exists(dataset_dir):
        print(r"""
            dataset "/data/dataset/llr_ood_genomics" does not exist
            download dataset from s3://public-datasets-contrib/Genomics_ood/
        """)
        raise FileNotFoundError(dataset_dir)
    if type == "train":
        x = load_pkl(os.path.join(dataset_dir, "before_2011_in_tr", "feature.pkl"))
        y = load_pkl(os.path.join(dataset_dir, "before_2011_in_tr", "label.pkl"))
    elif type == "ind_val":
        x = load_pkl(os.path.join(dataset_dir, "between_2011-2016_in_val", "feature.pkl"))
        y = load_pkl(os.path.join(dataset_dir, "between_2011-2016_in_val", "label.pkl"))
    elif type == "ood_val":
        x = load_pkl(os.path.join(dataset_dir, "between_2011-2016_ood_val", "feature.pkl"))
        y = load_pkl(os.path.join(dataset_dir, "between_2011-2016_ood_val", "label.pkl"))
    elif type == "ind_test":
        x = load_pkl(os.path.join(dataset_dir, "after_2016_in_test", "feature.pkl"))
        y = load_pkl(os.path.join(dataset_dir, "after_2016_in_test", "label.pkl"))
    elif type == "ood_test":
        x = load_pkl(os.path.join(dataset_dir, "after_2016_ood_test", "feature.pkl"))
        y = load_pkl(os.path.join(dataset_dir, "after_2016_ood_test", "label.pkl"))
    else:
        raise NotImplementedError

    np.random.seed(42)
    random_idx = np.random.permutation(list(range(len(x))))[:50000]
    x = x[random_idx]
    y = y[random_idx]
    x = torch.Tensor(x)
    y = torch.LongTensor(y)
    dataset = data.TensorDataset(x, y)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, num_workers=8)
    return dataloader
