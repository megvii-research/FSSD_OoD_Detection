import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from lib.model.facerec import l2_norm, one_hot

class AMSoftmax(nn.Module):
    def __init__(self, num_classes, embedding_size=512,  s=64.0, m=0.35):
        super(AMSoftmax, self).__init__()
        self.num_classes = num_classes
        self.kernel = nn.Parameter(torch.Tensor(embedding_size, num_classes))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m  # the margin value, default is 0.5
        self.s = s  # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m  # issue 1
        self.threshold = math.cos(math.pi - m)

    def get_logit(self, embeddings):
        kernel_norm = l2_norm(self.kernel, dim=0)
        embeddings = embeddings * self.s
        logit = torch.mm(embeddings, kernel_norm)
        return logit

    def forward(self, embeddings, label=None, istraining=True):
        """ CosFace Loss
        """
        if istraining == True:
            assert label is not None
            kernel_norm = l2_norm(self.kernel, dim=0)
            embeddings = embeddings * self.s
            logit = torch.mm(embeddings, kernel_norm)
            s_m = self.s * self.m
            gt_one_hot = one_hot(label, num_classes=self.num_classes, on_value=s_m, off_value=0.0)
            output = logit - gt_one_hot
            return output
        else:
            assert label is None
            return self.get_logit(embeddings)
