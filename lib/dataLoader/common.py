import os
import sys
import collections
import pickle
from concurrent import futures
import time
from multiprocessing import Process, Queue
import PIL

import torch
import numpy as np
import nori2
import cv2
# from meghair.utils.imgproc import imdecode
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

fetcher = nori2.Fetcher()


def imdecode(data, *, require_chl3=True, require_alpha=False):
    """decode images in common formats (jpg, png, etc.)

    :param data: encoded image data
    :type data: :class:`bytes`
    :param require_chl3: whether to convert gray image to 3-channel BGR image
    :param require_alpha: whether to add alpha channel to BGR image

    :rtype: :class:`numpy.ndarray`
    """
    img = cv2.imdecode(np.fromstring(data, np.uint8), cv2.IMREAD_UNCHANGED)
    def _gif_decode(data):
        try:
            import io
            from PIL import Image

            im = Image.open(io.BytesIO(data))
            im = im.convert('RGB')
            return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        except Exception:
            return

    if img is None and len(data) >= 3 and data[:3] == b'GIF':
        # cv2 doesn't support GIF, try PIL
        img = _gif_decode(data)

    assert img is not None, 'failed to decode'
    if img.ndim == 2 and require_chl3:
        img = img.reshape(img.shape + (1,))
    if img.shape[2] == 1 and require_chl3:
        img = np.tile(img, (1, 1, 3))
    if img.ndim == 3 and img.shape[2] == 3 and require_alpha:
        assert img.dtype == np.uint8
        img = np.concatenate([img, np.ones_like(img[:, :, :1]) * 255], axis=2)
    return img

def read_one_image(img_key, img_size):
    img_ldmks_bytes = fetcher.get(img_key)
    try:
        img_bytes = pickle.loads(img_ldmks_bytes)['img']
        img = imdecode(img_bytes)
    except Exception:
        img = imdecode(img_ldmks_bytes)

    if img_size > 0:
        img = cv2.resize(img, (img_size, img_size))
    # img = img.transpose(2, 0, 1)
    if img.shape[2] != 3:
        img = img[:,:,:3]
    img = img.astype('uint8')
    img = PIL.Image.fromarray(img)
    return img

class _Dataset(Dataset):
    def __init__(self, align5p_nori_id, img_size, transform, labels=None):
        self.align5p_nori_id = align5p_nori_id
        self.img_size = img_size
        # self.transform = lambda img: img.astype('float32')  # add augmentation here
        self.transform = transform
        self.labels = labels
        
    def __getitem__(self, idx):
        if self.labels is None:
            return self.transform(read_one_image(self.align5p_nori_id[idx], self.img_size))
        else:
            return self.transform(read_one_image(self.align5p_nori_id[idx], self.img_size)), self.labels[idx]

    def __len__(self):
        return len(self.align5p_nori_id)

class InferenceDataLoader(DataLoader):
    def __init__(self, align5p_nori_id, img_size, transform, labels=None, shuffle=False, batch_size=512, num_workers=8):
        dataset = _Dataset(align5p_nori_id, img_size, transform, labels)
        super(InferenceDataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=num_workers)
