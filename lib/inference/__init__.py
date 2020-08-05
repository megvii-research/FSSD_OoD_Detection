import os
import sys
import pickle

from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def get_feature_dim_list(model, img_size, inp_channel, flat=True):
    model.eval()
    model = model.cuda()
    if img_size != -1:
        temp_x = torch.rand(2, inp_channel, img_size, img_size).cuda()
    else:
        temp_x = torch.randint(0,4,(2,250)).cuda()
    if flat:
        try:
            fc_out, temp_list = model.feature_list(temp_x)
        except:
            fc_out, temp_list = model.module.feature_list(temp_x) # data parallel object
    else:
        try:
            fc_out, temp_list = model.nonflat_feature_list(temp_x)
        except:
            fc_out, temp_list = model.module.nonflat_feature_list(temp_x) # data parallel object
    num_output = len(temp_list)
    feature_dim_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_dim_list[count] = out.size(1)
        count += 1
    num_classes = fc_out.size(1)
    return feature_dim_list, num_classes
