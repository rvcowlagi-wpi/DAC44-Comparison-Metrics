# ========== Dependencies
import torch
import torch.nn as nn
import numpy
import pandas
from torch.autograd.variable import Variable

device = "cuda"

# ===== Problem size
nFeatures = 200
nTimeStamps = 100
dt_ = 0.1
dimLatent = 2
myBatchSize = 30


# ========== Encoder model
dimEnc100 = 512
dimEnc200 = 1024
dimEnc300 = 512
dimEnc400 = 256
dimEnc500 = 64

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(nFeatures, dimEnc100),  # input layer
            nn.ELU(),
            nn.Linear(dimEnc100, dimEnc200),  # hidden layer 1
            nn.ELU(),
            nn.Linear(dimEnc200, dimEnc300),  # hidden layer 2
            nn.ELU(),
            nn.Linear(dimEnc300, dimEnc400),  # hidden layer 3
            nn.ELU(),
            nn.Linear(dimEnc400, dimEnc500),  # hidden layer 4
            nn.ELU(),
            nn.Linear(dimEnc500, dimLatent), # output layer, same dim as generator latent space
        )

    def forward(self, x):
        return self.layers(x)


# ========== Decoder model, same as the generator model
dimDec100 = 64
dimDec200 = 256
dimDec300 = 512
dimDec400 = 1024
dimDec500 = 512


class DecoderGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(dimLatent, dimDec100),  # input layer
            nn.ELU(),
            nn.Linear(dimDec100, dimDec200),  # hidden layer 1
            nn.ELU(),
            nn.Linear(dimDec200, dimDec300),  # hidden layer 2
            nn.ELU(),
            nn.Linear(dimDec300, dimDec400),  # hidden layer 3
            nn.ELU(),
            nn.Linear(dimDec400, dimDec500),  # hidden layer 4
            nn.ELU(),
            nn.Linear(dimDec500, nFeatures), # output layer
        )

    def forward(self, x):
        return self.layers(x)


# ========== Data-driven discriminator model
dimDisc100 = 512
dimDisc200 = 256
dimDisc300 = 128
dimDisc400 = 64
dimDisc500 = 16

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(nFeatures, dimDisc100),  # input layer
            nn.ELU(),
            nn.Linear(dimDisc100, dimDisc200),  # hidden layer 1
            nn.ELU(),
            nn.Linear(dimDisc200, dimDisc300),  # hidden layer 2
            nn.ELU(),
            nn.Linear(dimDisc300, dimDisc400),  # hidden layer 3
            nn.ELU(),
            nn.Linear(dimDisc400, dimDisc500),  # hidden layer 4
            nn.ELU(),
            nn.Linear(dimDisc500, 1),
            nn.Sigmoid(),  # output layer
        )

    def forward(self, x):
        return self.layers(x)


# ========== Rules discriminator: straight lines
tmp1 = torch.arange(0, nFeatures, 1)
tmp2 = torch.ones(nFeatures)
C = torch.vstack((tmp1, tmp2))
C = torch.transpose(C, 0, 1)


def rules_d_straight_line(y_, device):
    C.to(device)
    x_ = torch.linalg.lstsq(C, y_).solution
    return torch.norm(y_ - torch.matmul(C, x_))


# =========== Rules discriminator: LTI state model
def rules_d_state1(y_, device_):
    x1_ = y_[:, 0:nTimeStamps]
    x2_ = y_[:, nTimeStamps:nFeatures]

    penalty_ = torch.zeros([myBatchSize, nFeatures]).to(device_)  # this is just for proper sizing

    for m in range(1, nTimeStamps, 1):

        penalty_[:, m] = (10 * (x1_[:, m] - x1_[:, m - 1]) - x2_[:, m - 1])  # 10 = 1/dt_
        penalty_[:, m + nTimeStamps] = (10 * (x2_[:, m] - x2_[:, m - 1]) + 5 * x1_[:, m - 1])

    return penalty_


# ========== Label for fake and real data"""
def ones_target(size):
    #   Tensor containing ones, with shape = size    """
    data = Variable(torch.ones(size, 1)).to(device)
    return data


def zeros_target(size):
    #   Tensor containing zeros, with shape = size    """
    data = Variable(torch.zeros(1, size)).to(device)
    return data


# ========== Load the real training data
data2d = numpy.array(pandas.read_csv('Data/data2D.txt'))
trainLoader = torch.utils.data.DataLoader(data2d, batch_size=myBatchSize)

