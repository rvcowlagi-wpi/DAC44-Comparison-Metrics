import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.autograd.variable import Variable
import matplotlib.pyplot as plt
import time

# ===== Problem size
nFeatures = 200
dimLatent = 16
nTimeStamps = 100
dt_ = 0.1
myBatchSize = 1

# ======== Load the real training data
data_2d = np.array(pd.read_csv('Data/data_2d.txt'))
train_loader = torch.utils.data.DataLoader(data_2d, batch_size=my_batch_size)

# ========== Encoder model
dimEncInput = nFeatures
dimEnc100 = 512
dimEnc200 = 1024
dimEnc300 = 512
dimEnc400 = 256
dimEnc500 = 64
dimEncOutput = dimLatent  # encoder output dimension should be the same as generator latent space dimension

# ===== Setup of Encoder
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(dimEncInput, dimEnc100),  # input layer
            nn.ELU(),
            nn.Linear(dimEnc100, dimEnc200),  # hidden layer 1
            nn.ELU(),
            nn.Linear(dimEnc200, dimEnc300),  # hidden layer 2
            nn.ELU(),
            nn.Linear(dimEnc300, dimEnc400),  # hidden layer 3
            nn.ELU(),
            nn.Linear(dimEnc400, dimEnc500),  # hidden layer 4
            nn.ELU(),
            nn.Linear(dimEnc500, dimEncOutput), # output layer
        )

    def forward(self, x):
        return self.layers(x)


# ========== Decoder model
dimDecInput = dimLatent
dimDec100 = 64
dimDec200 = 256
dimDec300 = 512
dimDec400 = 1024
dimDec500 = 512
dimDecOutput = nFeatures

# ===== Setup of Encoder
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(dimDecInput, dimDec100),  # input layer
            nn.ELU(),
            nn.Linear(dimDec100, dimDec200),  # hidden layer 1
            nn.ELU(),
            nn.Linear(dimDec200, dimDec300),  # hidden layer 2
            nn.ELU(),
            nn.Linear(dimDec300, dimDec400),  # hidden layer 3
            nn.ELU(),
            nn.Linear(dimDec400, dimDec500),  # hidden layer 4
            nn.ELU(),
            nn.Linear(dimDec500, dimDecOutput), # output layer
        )

    def forward(self, x):
        return self.layers(x)