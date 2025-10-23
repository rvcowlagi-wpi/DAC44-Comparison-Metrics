import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from torch.autograd.variable import Variable
import matplotlib.pyplot as plt
import time

# DIFFERENCE FROM V01
# This version has more hidden layers

device = "cuda"

start_clock = time.time()

n_features = 175
n_discretization = 25
my_batch_size = 2000

data_mintime = np.array(pd.read_csv('Data/data_mintime.txt'))
train_loader = torch.utils.data.DataLoader(data_mintime, batch_size=my_batch_size)

# Trajectory endpoints; these are fixed in the dataset
x_init = [0.0, 0.8]
x_term = [-0.8, -0.9]

# STEP 3A: Create generator model class
input_dimG = 20
dimG_1 = 64
dimG_2 = 100
dimG_3 = 225
dimG_4 = 400
dimG_5 = 625
dimG_6 = 900


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dimG, dimG_1),  # input layer
            nn.ReLU(),
            nn.Linear(dimG_1, dimG_2),  # hidden layer 1
            nn.ReLU(),
            nn.Linear(dimG_2, dimG_3),  # hidden layer 2
            nn.ReLU(),
            nn.Linear(dimG_3, dimG_4),  # hidden layer 3
            nn.ReLU(),
            nn.Linear(dimG_4, dimG_5),  # hidden layer 4
            nn.ReLU(),
            nn.Linear(dimG_5, dimG_6),  # hidden layer 5
            nn.ReLU(),
            # nn.Linear(1024, 2048),  # hidden layer 6
            # nn.ReLU(),
            nn.Linear(dimG_6, n_features)  # output layer
        )

    def forward(self, x):
        return self.layers(x)


# STEP 3B: Create discriminator model class
dimD_1 = 900
dimD_2 = 625
dimD_3 = 400
dimD_4 = 225
dimD_5 = 100
dimD_6 = 25


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(n_features, dimD_1),  # input layer
            nn.ReLU(),
            nn.Linear(dimD_1, dimD_2),  # hidden layer 1
            nn.ReLU(),
            nn.Linear(dimD_2, dimD_3),  # hidden layer 2
            nn.ReLU(),
            nn.Linear(dimD_3, dimD_4),  # hidden layer 3
            nn.ReLU(),
            nn.Linear(dimD_4, dimD_5),  # hidden layer 4
            nn.ReLU(),
            # nn.Linear(64, 32),  # hidden layer 5
            # nn.ReLU(),
            nn.Linear(dimD_5, 1),  # hidden layer 6
            nn.Sigmoid()  # output layer
        )

    def forward(self, x):
        return self.layers(x)


# STEP 4: Instantiate generator and discriminator classes
modelG = Generator()
modelD = Discriminator()
modelG.to(device)
modelD.to(device)

# STEP 5: Instantiate loss and optimizer
criterion = nn.SmoothL1Loss().to(device)  # nn.BCEWithLogitsLoss()  # nn.CrossEntropyLoss()
loss_dg = nn.MSELoss().to(device)

learning_rate_gen = 0.01
learning_rate_disc = 0.01
optimizerD = torch.optim.SGD(modelD.parameters(), lr=learning_rate_disc)
optimizerG = torch.optim.SGD(modelG.parameters(), lr=learning_rate_gen)

# Label for fake and real data"""
def ones_target(size):
    #   Tensor containing ones, with shape = size    """
    data = Variable(torch.ones(size, 1)).to(device)
    return data


def zeros_target(size):
    #   Tensor containing zeros, with shape = size    """
    data = Variable(torch.zeros(size, 1)).to(device)
    return data


n_epochs = 3000
n_iter = 0
G_losses = []
D_losses = []
for epoch in range(n_epochs):
    print('\n======== Epoch ', epoch, ' ========')
    for i, (traj) in enumerate(train_loader):
        # DISCRIMINATOR
        # Reset gradients
        optimizerD.zero_grad()
        modelD.zero_grad()

        # Train with real examples from dataset
        inputs = traj.float().to(device)

        y_d = modelD(inputs)

        # Calculate errorD and backpropagate
        errorD_real = loss_dg(y_d, ones_target(len(traj)))
        # errorD_real.backward()

        # Train with fake examples from generator
        z = torch.distributions.uniform.Uniform(-1, 1).sample([my_batch_size, input_dimG]).to(device)  # latent vector
        y_g = modelG(z)
        y_d_fake = modelD(y_g)

        # Calculate errorD and backpropagate
        errorD_fake = loss_dg(y_d_fake, zeros_target(my_batch_size))
        # errorD_fake.backward()

        # Compute errorD of D as sum over the fake and the real batches"""
        errorD = errorD_fake + errorD_real
        errorD.backward()

        # Update D
        optimizerD.step()

        D_losses.append(errorD.item())

stop_clock = time.time()
elapsed_hours = int((stop_clock - start_clock) // 3600)
elapsed_minutes = int((stop_clock - start_clock) // 60 - elapsed_hours * 60)
elapsed_seconds = (stop_clock - start_clock) - elapsed_hours * 3600 - elapsed_minutes * 60

print('\nElapsed time ' + str(elapsed_hours) + ':' + str(elapsed_minutes) + ':' + "%.2f" % elapsed_seconds)

# # STEP 8: Plot a few sample generator results
t = torch.linspace(0, 10, n_features).to("cpu")
fig1, ax1 = plt.subplots(3, 3)

px = 1 / plt.rcParams['figure.dpi']
fig1.set_figwidth(1800 * px)
fig1.set_figheight(1600 * px)
for m1 in range(0, 3):
    for m2 in range(0, 3):
        z = torch.distributions.uniform.Uniform(-1, 1).sample([1, input_dimG]).to(device)
        y = modelG(z).to("cpu")

        y_plt = y[0].detach().numpy()

        # y_plt_x1 = y_plt[(n_discretization):(2*n_discretization)]
        # y_plt_x2 = y_plt[(2*n_discretization):(3*n_discretization)]

        y_plt_x1 = y_plt[0:n_discretization]
        y_plt_x2 = y_plt[n_discretization:(2 * n_discretization)]

        ax1[m1, m2].plot(y_plt_x1, y_plt_x2)
        ax1[m1, m2].plot(x_init[0], x_init[1], marker="o", markersize=5)
        ax1[m1, m2].plot(x_term[0], x_term[1], marker="s", markersize=5)
        ax1[m1, m2].axis('equal')

        # y_plt_x1 = data_mintime[(3 * m1 + m2), 0:n_discretization]
        # y_plt_x2 = data_mintime[(3 * m1 + m2), n_discretization:(2 * n_discretization)]
        #
        # ax1[m1, m2].plot(y_plt_x1, y_plt_x2)
        # ax1[m1, m2].plot(x_init[0], x_init[1], marker="o", markersize=5)
        # ax1[m1, m2].plot(x_term[0], x_term[1], marker="s", markersize=5)
        # ax1[m1, m2].axis('equal')

# figName = "Results/2023-01-14-LTI/lti_1d_ep" + str(nEpochs) + "_ex" + \
#            str(len(train_loader.dataset) // nFeatures) + "_pt" + str(nFeatures) + ".png"
# figName = "Results/2023-01-14-LTI/lti_2d_ep" + str(nEpochs) + "_ex" + \
#            str(len(train_loader.dataset)) + "_pt" + str(nFeatures) + ".png"
# fig1.savefig(figName, bbox_inches='tight')


fig2 = plt.figure()
fig2.set_figwidth(1200 * px)
fig2.set_figheight(800 * px)

plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

#
