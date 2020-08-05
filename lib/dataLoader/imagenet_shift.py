import os
import sys

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from lib.utils.config import imagenet_shift_paths
from lib.utils.io import load_pkl
from lib.dataLoader.common import InferenceDataLoader

def getImageNetTrainDataSet(batch_size, transform):
    dataSetDir = "/data/datasets/imagenet_train"
    align5p_nori_id = load_pkl(os.path.join(dataSetDir, "nori_keys"))[0::10]
    labels = load_pkl(os.path.join(dataSetDir, "labels"))[0::10]
    dataloader = InferenceDataLoader(align5p_nori_id, -1, transform, labels=labels, batch_size=batch_size)
    return dataloader

def getImageNetTestDataSet(batch_size, transform):
    dataSetDir = "s3://yejinxing-bmks/datasets/imagenet-val/"
    align5p_nori_id = load_pkl(os.path.join(dataSetDir, "align5p.nori_id"))
    # dataloader = InferenceDataLoader(align5p_nori_id[0::5], -1, transform, batch_size=batch_size)
    dataloader = InferenceDataLoader(align5p_nori_id, -1, transform, batch_size=batch_size)
    return dataloader

def getImageNetShiftDataSet(shift_type, intensity, transform, batch_size=64):
    dataSetDir = imagenet_shift_paths[shift_type][intensity]
    align5p_nori_id = load_pkl(os.path.join(dataSetDir, "align5p.nori"))
    # dataloader = InferenceDataLoader(align5p_nori_id[0::5], -1, transform, batch_size=batch_size)
    dataloader = InferenceDataLoader(align5p_nori_id, -1, transform, batch_size=batch_size)
    return dataloader

