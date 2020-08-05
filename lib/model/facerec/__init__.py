import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def l2_norm(x, dim):
    norm = torch.norm(x, p=2, dim=dim, keepdim=True)
    # if norm > 0:
    x = x / norm
    return x

def one_hot(label, num_classes, on_value=1.0, off_value=0.0):
    res = torch.ones(label.shape[0], num_classes).to(label.device)*off_value
    res = res.scatter(dim=1, index=label.view(-1,1), value=on_value)
    return res

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

