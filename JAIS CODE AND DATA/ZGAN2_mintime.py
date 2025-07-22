# PROGRAM DESCRIPTION: This code trains the GAN model implementing the Hamiltonian and Heading Angle criterion for Zermelo Navigation problem

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
import csv
from matplotlib.ticker import ScalarFormatter

device = "cuda"
start_clock = time.time()

n_features = 175
n_discretization = 25
my_batch_size = 64

# STEP 1: Loading the data
data_mintime = np.array(pd.read_csv('C:/Users/nubapat/OneDrive - Worcester Polytechnic Institute (wpi.edu)/2024-JNL-NUBapat/Archives/Scatter-Plots/traindata_mintime_7states_4000_points_GAN.txt'))
train_loader = torch.utils.data.DataLoader(data_mintime, batch_size=my_batch_size)


# Trajectory parameters; these are fixed in the dataset
x_init = [0.0, 0.8]
x_term = [-0.8, -0.9]
V = 0.05

# STEP 2: Create Discriminator model class
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
            nn.Linear(50, dimD_1),  # input layer
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimD_1, dimD_2),  # hidden layer 1
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimD_2, dimD_3),  # hidden layer 2
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimD_3, dimD_4),  # hidden layer 3
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimD_4, dimD_5),  # hidden layer 4
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            # nn.Linear(64, 32),  # hidden layer 5
            # nn.ReLU(),
            nn.Linear(dimD_5, 1),  # hidden layer 6
            nn.Sigmoid()  # output layer
        )

    def forward(self, x):
        return self.layers(x)


# # #STEP 3: Create generator class with the Hamiltonian at the output
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
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimG_1, dimG_2),  # hidden layer 1
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimG_2, dimG_3),  # hidden layer 2
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimG_3, dimG_4),  # hidden layer 3
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimG_4, dimG_5),  # hidden layer 4
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimG_5, dimG_6),  # hidden layer 5
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            # nn.Linear(1024, 2048),  # hidden layer 6
            # nn.ReLU(),
            nn.Linear(dimG_6, n_features)  # output layer
        )

    def calculateHamiltonian(self, arr):
        H = list()
        Psi = list()

        for state in range(n_discretization):
            hamil = 1 + arr[:, state + 75] * V * torch.cos(arr[:,state + 50]) + arr[:, state + 75] * arr[:, state + 125] + arr[:, state + 100] * V * \
                torch.sin(arr[:, state + 50]) + arr[:, state + 100] * arr[:, state + 150]
            head = torch.atan(arr[:, state + 100]/arr[:, state + 75])
#
            H.append(hamil)
            Psi.append(head)
        H = torch.stack(H)
        Psi = torch.stack(Psi)
        return torch.transpose(H, 0, 1), torch.transpose(Psi, 0, 1)
#
#
#
#
    def forward(self, x):
        model_output = self.layers(x)
        hamil,head = self.calculateHamiltonian(model_output)
        return hamil,head,model_output

# STEP 4: Instantiate generator and discriminator classes
def main():
    modelG = Generator()
    modelD = Discriminator()
    modelD.to(device)
    modelG.to(device)
    modelG.train()
    modelD.train()
# STEP 5: Instantiate loss and optimizer
    loss_dg1 = nn.MSELoss().to(device)
    # loss_dg = nn.BCELoss().to(device)
    learning_rate_gen = 0.01
    learning_rate_disc = 0.01
    optimizerD = torch.optim.SGD(modelD.parameters(), lr=learning_rate_disc)
    optimizerG = torch.optim.SGD(modelG.parameters(), lr=learning_rate_gen)
    torch.manual_seed(13)


    n_epochs = 500
    n_iter = 0
    DISC_ITERATIONS = 1
    GEN_ITERATIONS = 1
    # G_losses = [] # Total generator loss
    GD_losses = [] # loss from Discriminator
    D_losses = [] # Discriminator loss
    MSE_losses = [] # Hamiltonian loss

    for epoch in range(n_epochs):

        print('\n======== Epoch ', epoch + 1, ' ========')
        for i, (traj) in enumerate(train_loader):
            for _ in range(DISC_ITERATIONS):
                optimizerD.zero_grad()
                modelD.zero_grad()

                # Train with real examples from dataset
                inputs = traj.float().to(device)
                # inputs = inputs[:,25:75]
                y_d = modelD(inputs[:,0:50])

                # Calculate errorD and backpropagate
                errorD_real = loss_dg1(y_d, torch.ones_like(y_d))

                # Train with fake examples from generator
                z = torch.distributions.uniform.Uniform(-1, 1).sample([my_batch_size, input_dimG]).to(device)  # latent vector
                _, _, y = modelG(z)
                yd = y[:, 0:50]
                y_d_fake = modelD(yd)

                # Calculate errorD and backpropagate
                errorD_fake = loss_dg1(y_d_fake, torch.zeros_like((y_d_fake)))

                # Compute errorD of D as sum over the fake and the real batches"""
                errorD = errorD_fake + errorD_real
                errorD.backward()

                # Update D
                optimizerD.step()
            D_losses.append(errorD.item())

            # Reset gradients of Generator
            for _ in range(GEN_ITERATIONS):
                modelG.zero_grad()
                optimizerG.zero_grad()
                z = torch.distributions.uniform.Uniform(-1, 1).sample([my_batch_size, input_dimG]).to(device)  # latent vector
                hamil,head, y_pred = modelG(z)
                yg = y_pred[:, 0:50]
                y_d_fakeG = modelD(yg)  # Since we just updated D, perform another forward pass of all-fake batch through D

                # Calculate errorD and backpropagate """
                errorG1 = loss_dg1(y_d_fakeG, torch.ones_like(y_d_fakeG))
                loss_MSE = loss_dg1(hamil, torch.zeros_like(hamil)) + loss_dg1(head, y_pred[:, 50:75])
                errorG = loss_MSE + errorG1
                errorG.backward()
                # Update G
                optimizerG.step()
            n_iter += 1
            GD_losses.append(errorG.item())
            avg_errorGD = sum(GD_losses) / len(GD_losses)
            MSE_losses.append(loss_MSE.item())
            avg_errorMSE = sum(MSE_losses) / len(MSE_losses)
            # D_losses.append(errorD.item())
            avg_errorD = sum(D_losses) / len(D_losses)

        print('Epoch [{}/{}], Discriminator Loss: {:.4f}'.format(epoch + 1, n_epochs, avg_errorD))
        print('Epoch [{}/{}], GD Loss: {:.4f}'.format(epoch + 1, n_epochs, avg_errorGD))
        print('Epoch [{}/{}], MSE Loss: {:.4f}'.format(epoch + 1, n_epochs, avg_errorMSE))

    torch.save(modelG.state_dict(), 'Generator_model_ZGAN2.pth')
    torch.save(modelD.state_dict(), 'Discriminator_model_ZGAN2.pth')


    # STEP 9: Generate 1000 samples using the trained generator
    num_samples = 1000  # Number of samples to generate
    latent_dim = input_dimG  # Dimension of latent vector

    print("\nGenerating 1000 samples with the trained generator...")
    generated_samples = []

    # Ensure the generator is in evaluation mode
    modelG.eval()

    # Generate samples in batches
    with torch.no_grad():
        for _ in range(num_samples // my_batch_size):
            z = torch.distributions.uniform.Uniform(-1, 1).sample([my_batch_size, latent_dim]).to(device)
            _, _, y_generated = modelG(z)
            generated_samples.append(y_generated.cpu().numpy())

        # Handle remaining samples if num_samples is not a multiple of batch_size
        remaining_samples = num_samples % my_batch_size
        if remaining_samples > 0:
            z = torch.distributions.uniform.Uniform(-1, 1).sample([remaining_samples, latent_dim]).to(device)
            _, _, y_generated = modelG(z)
            generated_samples.append(y_generated.cpu().numpy())

    # Combine all generated samples
    generated_samples = np.vstack(generated_samples)

    # Save the generated samples to a CSV file
    output_csv_path = 'C:/Users/nubapat/OneDrive - Worcester Polytechnic Institute (wpi.edu)/2024-JNL-NUBapat/Archives/Scatter-Plots/ZGAN2_test.csv'
    np.savetxt(output_csv_path, generated_samples, delimiter=',')

    print(f"Generated samples saved to {output_csv_path}")

if __name__ == "__main__":
    main()
