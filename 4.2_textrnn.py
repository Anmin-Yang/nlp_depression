"""
Architecture of TextRNN
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
class TextRnn(nn.Module):
    def __init__(self, embed_num = 1000, embed_dim = 300, max_words = 0):
        super(TextRnn, self).__init__()

        # self.embed = nn.Embedding(embed_num, embed_dim)
        self.max_words = max_words
        self.bidirectional = True
        self.lstm_dim = 64
        self.lstm_num_layer = 2
        self.lstm = nn.LSTM(300, self.lstm_dim, self.lstm_num_layer,
                            bidirectional=self.bidirectional, batch_first=True, dropout=0.5)
        self.maxpool = nn.MaxPool1d(self.max_words)
        if(self.bidirectional):
            self.fc = nn.Linear(self.lstm_dim * 2 + 300, 2)
        else:
            self.fc = nn.Linear(self.lstm_dim + 300, 2)

    def forward(self, x):
        # embed = self.embedding(x)
        out, _ = self.lstm(x)
        # print(out.shape)
        out = torch.cat((x, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        # print(out.shape)
        out = self.maxpool(out).squeeze()
        # print(out.shape)
        out = self.fc(out)
        return out

class TextRnn_att(nn.Module):
    def __init__(self, embed_num = 1000, embed_dim = 300):
        super(TextRnn_att, self).__init__()
        self.lstm_dim = 32
        self.lstm = nn.LSTM(300, self.lstm_dim, 2,
                            bidirectional=True, batch_first=True, dropout=0.5)
        self.tanh1 = nn.Tanh()
        # self.u = nn.Parameter(torch.Tensor(256 * 2, 256 * 2))
        self.w = nn.Parameter(torch.zeros(self.lstm_dim * 2))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(self.lstm_dim * 2, 64)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        # embed = self.embedding(x)
        H, _ = self.lstm(x)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]

        M = self.tanh1(H)  # [128, 32, 256]
        # M = torch.tanh(torch.matmul(H, self.u))
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1]
        out = H * alpha  # [128, 32, 256]
        out = torch.sum(out, 1)  # [128, 256]
        out = F.relu(out)
        out = self.fc1(out)
        out = self.fc(out)  # [128, 64]
        return out

if __name__=="__main__":
    model = TextRnn()
    data = torch.randn((10, 273, 300))
    output = model(data)
    print(output.shape)
