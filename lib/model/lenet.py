from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        # output = F.log_softmax(x, dim=1)
        return x
        
    def feature_list(self, x):
        out_list = []
        out = F.relu(self.conv1(x))
        out_list.append(out.contiguous().view(out.size(0), -1))
        out = F.relu(self.conv2(out))
        out_list.append(out.contiguous().view(out.size(0), -1))
        out = F.max_pool2d(out, 2)
        out = torch.flatten(out, 1)
        out = F.relu(self.fc1(out))
        out_list.append(out.contiguous().view(out.size(0), -1))
        out = self.fc2(out)
        out_list.append(out.contiguous().view(out.size(0), -1))
        y = F.softmax(out, dim=1)
        return y, out_list
    
    def nonflat_feature_list(self, x):
        """
        Used  in `test_maha`
        """
        out_list = []
        out = F.relu(self.conv1(x))
        out_list.append(out)
        out = F.relu(self.conv2(out))
        out_list.append(out)
        out = F.max_pool2d(out, 2)
        out = torch.flatten(out, 1)
        out = F.relu(self.fc1(out))
        out_list.append(out)
        out = self.fc2(out)
        out_list.append(out)
        y = F.softmax(out, dim=1)
        return y, out_list
    
    
    def mean_feat_list(self, x):
        out_list = []
        out = F.relu(self.conv1(x))
        out_list.append(out.contiguous().view(out.size(0), -1).mean(dim=0, keepdim=True))
        out = F.relu(self.conv2(out))
        out_list.append(out.contiguous().view(out.size(0), -1).mean(dim=0, keepdim=True))
        out = F.max_pool2d(out, 2)
        out = torch.flatten(out, 1)
        out = F.relu(self.fc1(out))
        out_list.append(out.contiguous().view(out.size(0), -1).mean(dim=0, keepdim=True))
        out = self.fc2(out)
        out_list.append(out.contiguous().view(out.size(0), -1).mean(dim=0, keepdim=True))
        return out_list

    # function to extact a specific feature
    def intermediate_forward(self, x, layer_index):
        x = self.conv1(x)
        out = F.relu(x)
        if layer_index == 1:
            out = F.relu(self.conv2(out))
        elif layer_index == 2:
            out = F.relu(self.conv2(out))
            out = F.max_pool2d(out, 2)
            out = torch.flatten(out, 1)
            out = F.relu(self.fc1(out))
        elif layer_index == 3:
            out = F.relu(self.conv2(out))
            out = F.max_pool2d(out, 2)
            out = torch.flatten(out, 1)
            out = F.relu(self.fc1(out))
            out = self.fc2(out) 
        return out