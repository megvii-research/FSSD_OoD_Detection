"""ResNet in PyTorch.
BasicBlock and Bottleneck module is from the original ResNet paper:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
PreActBlock and PreActBottleneck module is from the later paper:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
Original code is from https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
"""
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dropout_rate=0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        if dropout_rate is not 0.0:
            self.dropout = nn.Dropout2d(dropout_rate)
        else:
            self.dropout = lambda x: x

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            # no dropout for shortcut.
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(self.dropout(x))))
        out = self.bn2(self.conv2(self.dropout(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dropout_rate=0.0):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        if dropout_rate is not 0.0:
            self.dropout = nn.Dropout2d(dropout_rate)
        else:
            self.dropout = lambda x: x

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(self.dropout(out))
        out = self.conv2(self.dropout(F.relu(self.bn2(out))))
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, dropout_rate=0.0):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        if dropout_rate is not 0.0:
            self.dropout = nn.Dropout2d(dropout_rate)
        else:
            self.dropout = lambda x: x

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(self.dropout(x))))
        out = F.relu(self.bn2(self.conv2(self.dropout(out))))
        out = self.bn3(self.conv3(self.dropout(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBottleneck(nn.Module):
    """Pre-activation version of the original Bottleneck module."""

    expansion = 4

    def __init__(self, in_planes, planes, stride=1, dropout_rate=0.0):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        if dropout_rate is not 0.0:
            self.dropout = nn.Dropout2d(dropout_rate)
        else:
            self.dropout = lambda x: x

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(self.dropout(out))
        out = self.conv2(self.dropout(F.relu(self.bn2(out))))
        out = self.conv3(self.dropout(F.relu(self.bn3(out))))
        out += shortcut
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        num_classes=10,
        dropout_rate=0.0,
        method="dropout",
        target_dataset="cifar",
    ):
        """
        :param dropout_rate: always on dropout rate.
        :param method: dropout method.
        :param target_dataset: customized inputs settings for different datasets
        """
        super(ResNet, self).__init__()
        if (method in ["dropout", "ll_dropout", "dropout_nofirst"]) != (
            dropout_rate > 0.0
        ):
            raise ValueError(
                "Dropout rate should be nonzero iff a dropout method is used."
                "Method is {}, dropout is {}.".format(method, dropout_rate)
            )

        self.hidden_layer_dropout = (
            dropout_rate if method in ["dropout", "dropout_nofirst"] else 0.0
        )

        self.dropout_rate = dropout_rate
        self.method = method
        self.in_planes = 64

        # CIFAR-10, input size=(3, 32, 32)
        if target_dataset == "cifar10" or target_dataset == "svhn":
            self.conv1 = conv3x3(3, 64)
            self.max_pool = lambda x: x
        # image net 2012, input size=(3, 224, 224)
        elif target_dataset == "imagenet":
            self.conv1 = nn.Conv2d(
                3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        # ms1m, input size=(3, 112, 112)
        elif target_dataset == "ms1m":
            self.conv1 = nn.Conv2d(
                3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.max_pool = lambda x: x
        elif target_dataset == "mnist" or target_dataset == "fmnist" or target_dataset == "emnist":
            self.conv1 = nn.Conv2d(
                1, self.in_planes, kernel_size=1, stride=1, padding=3, bias=False
            )
            self.max_pool = lambda x: x

        self.bn1 = nn.BatchNorm2d(self.in_planes)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        if dropout_rate is 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = lambda x: x

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    stride,
                    dropout_rate=self.hidden_layer_dropout,
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):

        if (self.dropout_rate > 0.0) and (self.method != "dropout_nofirst"):
            x = nn.Dropout2d(self.hidden_layer_dropout)(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.max_pool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        y = self.linear(out)
        return y

    # function to extact the multiple features
    def feature_list(self, x):
        out_list = []
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.max_pool(out)
        out_list.append(out)
        out = self.layer1(out)
        out_list.append(out)
        out = self.layer2(out)
        out_list.append(out)
        out = self.layer3(out)
        out_list.append(out)
        out = self.layer4(out)
        out_list.append(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        y = self.linear(out)
        return y, out_list

    # function to extact a specific feature
    def intermediate_forward(self, x, layer_index):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.max_pool(out)
        if layer_index == 1:
            out = self.layer1(out)
        elif layer_index == 2:
            out = self.layer1(out)
            out = self.layer2(out)
        elif layer_index == 3:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
        elif layer_index == 4:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
        return out

    # function to extact the penultimate features
    def penultimate_forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.max_pool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        penultimate = self.layer4(out)
        out = F.avg_pool2d(penultimate, 4)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        y = self.linear(out)
        return y, penultimate


def ResNet18(num_c, dropout_rate=0.0, method="normal", target_dataset="cifar10"):
    return ResNet(
        PreActBlock,
        [2, 2, 2, 2],
        num_classes=num_c,
        dropout_rate=dropout_rate,
        method=method,
        target_dataset=target_dataset,
    )


def ResNet34(num_c, dropout_rate=0.0, method="normal", target_dataset="cifar10"):
    return ResNet(
        BasicBlock,
        [3, 4, 6, 3],
        num_classes=num_c,
        dropout_rate=dropout_rate,
        method=method,
        target_dataset=target_dataset,
    )


def ResNet50(num_c, dropout_rate=0.0, method="normal", target_dataset="cifar10"):
    return ResNet(
        Bottleneck,
        [3, 4, 6, 3],
        num_classes=num_c,
        dropout_rate=dropout_rate,
        method=method,
        target_dataset=target_dataset,
    )


def ResNet101(dropout_rate=0.0, method="normal", target_dataset="cifar10"):
    return ResNet(
        Bottleneck,
        [3, 4, 23, 3],
        dropout_rate=dropout_rate,
        method=method,
        target_dataset=target_dataset,
    )


def ResNet152(dropout_rate=0.0, method="normal", target_dataset="cifar10"):
    return ResNet(
        Bottleneck,
        [3, 8, 36, 3],
        dropout_rate=dropout_rate,
        method=method,
        target_dataset=target_dataset,
    )
