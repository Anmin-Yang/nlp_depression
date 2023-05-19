"""
Architecture of TextCNN
"""
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import torchvision.datasets as dsets
import torchvision.transforms as transforms

######### model textcnn
class Mish(nn.Module):
    def __init__(self):
        super().__init__()
        print("Mish activation loaded...")
    def forward(self,x):
        x = x * (torch.tanh(F.softplus(x)))
        return x
class TextCnn(nn.Module):
    def __init__(self, embed_num = 1000, embed_dim = 300, class_num = 2, kernel_num = 1, kernel_sizes = [k for k in range(2,3)], dropout = 0.5):
        super(TextCnn, self).__init__()

        Ci = 1
        Co = kernel_num
        
        # self.finetune_embedding = nn.Linear(embed_dim, embed_dim)
        # self.act = nn.ReLU()
        self.act = Mish()
        self.embed = nn.Embedding(embed_num, embed_dim)
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (f, embed_dim), padding = (2, 0)) for f in kernel_sizes])
        # self.bns1 = nn.ModuleList([nn.BatchNorm2d(Co) for f in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(Co * len(kernel_sizes), class_num)

    def forward(self, x):
        # x = self.embed(x)  # (N, token_num, embed_dim)
        # print(type(x))
        x = x.unsqueeze(1)  # (N, Ci, token_num, embed_dim)
        x = [self.act(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, token_num) * len(kernel_sizes)]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co) * len(kernel_sizes)]
        x = torch.cat(x, 1) # (N, Co * len(kernel_sizes))
        x = self.dropout(x)  # (N, Co * len(kernel_sizes))
        logit = self.fc(x)  # (N, class_num)
        return logit
class TextCnn_BN(nn.Module):
    def __init__(self, embed_num = 1000, embed_dim = 300, class_num = 2, kernel_num = 16, kernel_sizes = [k for k in range(2,8)], dropout = 0.5):
        super(TextCnn_BN, self).__init__()

        Ci = 1
        Co = kernel_num
        # self.act = nn.ReLU()
        self.act = Mish()
        self.embed = nn.Embedding(embed_num, embed_dim)
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (f, embed_dim), padding = (2, 0)) for f in kernel_sizes])
        self.bns1 = nn.ModuleList([nn.BatchNorm2d(Co) for f in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(Co * len(kernel_sizes), class_num)

    def forward(self, x):
        # x = self.embed(x)  # (N, token_num, embed_dim)
        # print(type(x))
        x = x.unsqueeze(1)  # (N, Ci, token_num, embed_dim)
        x = [self.act(bn(conv(x))).squeeze(3) for conv, bn in zip(self.convs1, self.bns1)]  # [(N, Co, token_num) * len(kernel_sizes)]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co) * len(kernel_sizes)]
        x = torch.cat(x, 1) # (N, Co * len(kernel_sizes))
        x = self.dropout(x)  # (N, Co * len(kernel_sizes))
        logit = self.fc(x)  # (N, class_num)
        return logit
 