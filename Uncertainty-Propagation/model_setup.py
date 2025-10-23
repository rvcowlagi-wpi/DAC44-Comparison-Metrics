# ========== Dependencies
import torch
import torch.nn as nn


device = "cuda"

# ===== Problem size
nStates = 2
nControl = 1
nFeatures = nStates + nControl
dimLatent = 2
dimOutput = nStates
myBatchSize = 30

# ========== Encoder
dimEnc100 = 16
dimEnc200 = 4


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Just a two layer network to encode
        # \xi = (x, u) to mean and variance
        self.layers = nn.Sequential(
            nn.Linear(nFeatures, dimEnc100),  # input layer
            nn.ELU(),
            nn.Linear(dimEnc100, dimEnc200),  # hidden layer 1
            nn.ELU(),
        )

        self.mean = nn.Linear(dimEnc200, dimLatent)
        self.log_var = nn.Linear(dimEnc200, dimLatent)

        self.training = True  # <=== No idea what this does

    def forward(self, x):
        hiddenState = self.layers(x)
        hiddenMean = self.mean(hiddenState)
        hiddenLogVar = self.mean(hiddenState)

        return hiddenMean, hiddenLogVar


# ========== Decoder (same as generator)
dimDec100 = 4
dimDec200 = 16


class DecoderGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(dimLatent, dimDec100),  # input layer
            nn.ELU(),
            nn.Linear(dimDec100, dimDec200),  # hidden layer 1
            nn.ELU(),
            nn.Linear(dimDec200, dimOutput),  # output layer
        )

    def forward(self, x):
        return self.layers(x)


# ========== VP model
class Model(nn.Module):
    def __init__(self, theEncoder, theDecoder):
        super().__init__()
        self.encoder = theEncoder
        self.decoder = theDecoder

    def reparametrization(self, mean_, var_):
        epsilon = torch.randn_like(var).to(device)  # sampling epsilon
        z = mean_ + var_ * epsilon  # reparametrization trick
        return z

    def forward(self, x):
        hiddenMean, hiddenLogVar = self.encoder(x)
        z = self.reparameterization(mean_, torch.exp(0.5 * logVar_))  # takes exponential function (log var -> var)
        xKPlus1 = self.decoder(z)

        return xKPlus1, hiddenMean, hiddenLogVar


weightLossR = 1     # weight for reconstruction loss
weightLossK = 1     # weight for regularization KL divergence loss


# ========== Loss function
def loss_function(x, xKPlus1, hiddenMean, hiddenLogVar):
    # x is target (training data)
    # xkPlus1 is decoder output
    lossReproduction = nn.functional.mse_loss(xKPlus1, x)
    lossRegularizationKLD = - 0.5 * torch.sum(1 + hiddenLogVar - hiddenMean.pow(2) - hiddenLogVar.exp())

    return weightLossR * lossReproduction + weightLossK * lossRegularizationKLD


# ========== Load the real training data
data2d = numpy.array(pandas.read_csv('Data/data2D.txt'))
trainLoader = torch.utils.data.DataLoader(data2d, batch_size=myBatchSize)
