import os, sys
import pickle
import numpy as np
from glob import glob
from tqdm import tqdm
import torch
import torch.nn as nn

from loguru import logger

# Bidirectional recurrent neural network (many-to-one)
class BiRNN(nn.Module):
    def __init__(self, vocab_size=4, embedding_dim=4, hidden_dim=512, num_layers=2, num_classes=10, drop_prob=0.1):
        super(BiRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=drop_prob, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim*2, num_classes)  # 2 for bidirection

    def forward(self, x):
        batch_size = x.size(0)
        x = x.long()
        embeds = self.embedding(x)

        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_dim).to(x.device) # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_dim).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(embeds, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        # Decode the hidden state of the last time step
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

    def feature_list(self, x):
        out_list = []        
        batch_size = x.size(0)
        x = x.long()
        embeds = self.embedding(x)
        out_list.append(embeds)

        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_dim).to(x.device) # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_dim).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(embeds, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        # Decode the hidden state of the last time step
        out_list.append(out[:,100,:])
        out_list.append(out[:,150,:])
        out_list.append(out[:,200,:])

        out = self.dropout(out[:, -1, :])
        out_list.append(out)
        out = self.fc(out)
        return out, out_list

    # function to extact a specific feature
    def intermediate_forward(self, x, layer_index):
        batch_size = x.size(0)
        x = x.long()
        out = self.embedding(x)
        if layer_index == 0:
            return out
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_dim).to(x.device) # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_dim).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(out, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        if layer_index == 1:
            out = out[:,100,:]
        elif layer_index == 2:
            out = out[:,150,:]
        elif layer_index == 3:
            out = out[:,200,:]
        elif layer_index == 4:
            out = self.dropout(out[:, -1, :])
        return out

