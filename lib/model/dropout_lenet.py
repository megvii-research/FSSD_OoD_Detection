import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, num_c=10, method="dropout", dropout_rate=0.0, inp_size=28):
        super(LeNet, self).__init__()
        if dropout_rate == 0:
            self.dropout_fn = self.dropout_fn_last = lambda x: x
        elif method == "dropout":
            self.dropout_fn = self.dropout_fn_last = nn.Dropout(dropout_rate)
        else:
            raise NotImplementedError
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, bias=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, bias=True)
        self.max_pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(9216, 128)
        self.fc_last = nn.Linear(128, num_c)

    def forward(self, x):
        x = F.relu(self.conv1(self.dropout_fn(x)))
        x = F.relu(self.conv2(self.dropout_fn(x)))
        x = self.max_pool(x)
        x = self.dropout_fn(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout_fn_last(x)
        x = self.fc_last(x)
        # x = F.log_softmax(x, dim=1)
        return x
