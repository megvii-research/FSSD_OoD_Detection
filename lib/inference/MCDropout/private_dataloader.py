import io

from PIL import Image
from torchvision.datasets import VisionDataset
import pandas as pd
import nori2 as nori
from brainpp.oss import OSSPath


def pil_loader(data):
    img = Image.open(io.BytesIO(data))
    return img.convert("RGB")


class ImageNet(VisionDataset):
    """`ImageNet <http://image-net.org/>`_ 2012 Classification Dataset.

    Mostly compatible with the conterpart in torchvision 0.4.1, except for a few
    removed attributes and arguments.

    Args:
        root (string): Root directory of the ImageNet Dataset.
        split (string, optional): The dataset split, supports ``train``, or ``val``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

     Attributes:
        targets (array-like): The class_index value for each image in the dataset
    """

    def __init__(
        self,
        root="s3://public-datasets-contrib/ILSVRC2012/processed/nori",
        split="train",
        # **kwargs,
    ):
        super().__init__(self, None, **kwargs)
        self.root = root
        self.split = split
        self.loader = pil_loader
        self._meta = pd.read_csv(
            (OSSPath(root) / "imagenet.{}.nori.csv".format(split)).open("rb")
        )
        self._fetcher = nori.Fetcher()

    @property
    def targets(self):
        return self._meta.class_id

    @property
    def filenames(self):
        return self._meta.filename

    @property
    def nori_keys(self):
        return self._meta.data_id

    @property
    def class_num(self):
        return len(set(self.targets))

    def __len__(self):
        return len(self.nori_keys)

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class
                and sample is whatever returned by transform (or a PIL image if
                transform is not given).
        """
        target = self.targets[idx]
        img_bytes = self._fetcher.get(self.nori_keys[idx])
        img = self.loader(img_bytes)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        # return img, target
        return np.array(img).transpose([1, 2, 0]), target

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_fetcher"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._fetcher = nori.Fetcher()


import os
import sys
import collections
import pickle
from concurrent import futures
import time
from multiprocessing import Process, Queue

import torch
import numpy as np
import nori2
import cv2
from meghair.utils.imgproc import imdecode


fetcher = nori2.Fetcher()
inpsize = 112


def read_one_image(img_key):
    img_ldmks_bytes = fetcher.get(img_key)
    img_bytes = pickle.loads(img_ldmks_bytes)["img"]
    img = imdecode(img_bytes)
    img = cv2.resize(img, (inpsize, inpsize))  # (112, 112, 3)
    img = img.transpose(2, 0, 1)
    img = img.astype("uint8")
    return img


def read_one_image_with_aug(img_key):
    from augmentor.preprocess import single_img_processor, stable_rng
    from augmentor.dataset_utils import landmark81_dict2nparray

    rng = stable_rng(stable_rng)
    img_ldmks_bytes = fetcher.get(img_key)
    img_ldmks = pickle.loads(img_ldmks_bytes)
    img_bytes = img_ldmks["img"]
    img = imdecode(img_bytes)
    # ===== aug
    ld = landmark81_dict2nparray(img_ldmks["ld"])
    img, _, _ = single_img_processor(img, ld, set(), rng)
    # =====
    img = cv2.resize(img, (inpsize, inpsize))  # (112, 112, 3)
    img = img.transpose(2, 0, 1)
    img = img.astype("uint8")
    return img


executor = futures.ThreadPoolExecutor(max_workers=64)


def read_images_from_imgkeys(img_keys, use_aug):
    # with futures.ThreadPoolExecutor(max_workers=32) as executor:
    if use_aug:
        res = executor.map(read_one_image_with_aug, img_keys)
    else:
        res = executor.map(read_one_image, img_keys)
    data = list(res)
    return np.array(data)


def read_imgs(align5p_nori_id, gt_info, batchsize, q, use_aug):
    """sampling by PID"""
    train_size = align5p_nori_id.shape[0]
    # train_index = np.array(range(train_size))

    clusters = []
    train_labels = np.zeros(train_size, dtype="int32")
    for i, (s, e) in enumerate(gt_info):
        train_labels[s:e] = i
        clusters.append(list(range(s, e)))

    while True:
        random_data = os.urandom(4)
        seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(seed)

        clusters = np.random.permutation(clusters)
        batch_clusters = clusters[:batchsize]
        img_idxs = []
        for clst in batch_clusters:
            idx = np.random.choice(len(clst), 1, replace=False)[0]
            img_idxs.append(clst[idx])
        imgkeys = align5p_nori_id[img_idxs]
        rawimgs = read_images_from_imgkeys(imgkeys, use_aug)
        labels = train_labels[img_idxs]
        q.put((rawimgs, labels))


# ===============


class TrainingDataLoader(object):
    def __init__(self, align5p_nori_id, gt_info, batchsize, use_aug=False):
        self.align5p_nori_id = align5p_nori_id
        self.gt_info = gt_info
        self.batchsize = batchsize
        self.use_aug = use_aug
        self.q = Queue(maxsize=256)
        self.workers = [
            Process(
                target=read_imgs,
                args=(
                    self.align5p_nori_id,
                    self.gt_info,
                    self.batchsize,
                    self.q,
                    self.use_aug,
                ),
            )
            for _ in range(8)
        ]

    def start_worker(self):
        for w in self.workers:
            w.start()

    def get_batch(self):
        return self.q.get()

    def terminate(self):
        for w in self.workers:
            w.terminate()
