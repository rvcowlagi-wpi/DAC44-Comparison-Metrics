import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn. metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from torch.autograd.variable import Variable
import matplotlib.pyplot as plt
import time

device = "cpu"

start_clock = time.time()

# Step 1: Load Dataset
# Step 2: Make Dataset Iterable
# Step 3: Create Model Class (G and D)
# Step 4: Instantiate Model Class
# Step 5: Instantiate Loss Class
# Step 6: Instantiate Optimizer Class
# Step 7: Train Model
# Step 8: View results: losses, plot of trajectory generated


# STEP 2: Make dataset iterable and load
#
# We have N training examples (each trajectory example).
# Split them into smaller batch sizes, say each batch with M examples, M < N.
# Then we have K iterations = N / M. An epoch consists of these 5 iterations.
# We can choose the number of epochs.

n_traj_pts = 100
my_batch_size = 1

data_1d = np.array(pd.read_csv('Data/data_1d.txt'))
data_1d_test = np.array(pd.read_csv('Data/data_1d_test.txt'))
# data2D = np.array(pd.read_csv('Data/data2D.txt'))

#dataset = torch.utils.data.TensorDataset(torch.Tensor(data_1d))
train_loader = torch.utils.data.DataLoader(data_1d, batch_size=my_batch_size)
test_loader = torch.utils.data.DataLoader(data_1d_test, batch_size=my_batch_size)

# STEP 3A: Create generator model class
class Generator(nn.Module):
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, hidden4_dim, output_dim):
        super(Generator, self).__init__()

        self.ReLU = nn.LeakyReLU()

        # Input layer
        self.fc1 = nn.Linear(input_dim, hidden1_dim)

        # Hidden layer 1
        self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)

        # Hidden layer 2
        self.fc3 = nn.Linear(hidden2_dim, hidden3_dim)

        # Hidden layer 3
        self.fc4 = nn.Linear(hidden3_dim, hidden4_dim)

        # Output layer
        self.fc5 = nn.Linear(hidden4_dim, output_dim)

    def forward(self, x):
        # Input layer
        out = self.fc1(x)
        out = self.ReLU(out)

        # Hidden layer 1
        out = self.fc2(out)
        out = self.ReLU(out)

        # Hidden layer 2
        out = self.fc3(out)
        out = self.ReLU(out)

        # Hidden layer 2
        out = self.fc4(out)
        out = self.ReLU(out)

        # Output layer
        out = self.fc5(out)

        return out


# STEP 3B: Create discriminator model class
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, output_dim):
        super(Discriminator, self).__init__()

        self.ReLU = nn.ReLU()

        # Input layer
        self.fc1 = nn.Linear(input_dim, hidden1_dim)

        # Hidden layer 1
        self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)

        # Hidden layer 2
        self.fc3 = nn.Linear(hidden2_dim, hidden3_dim)

        # Hidden layer 3
        self.fc4 = nn.Linear(hidden3_dim, output_dim)

        # Output layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input layer
        out = self.fc1(x)
        out = self.ReLU(out)

        # Hidden layer 1
        out = self.fc2(out)
        out = self.ReLU(out)

        # Hidden layer 2
        out = self.fc3(out)
        out = self.ReLU(out)

        # Hidden layer 3
        out = self.fc4(out)

        # Output layer
        out = self.sigmoid(out)
        return out


def setofrules(y_gen):

    solution_found_ = 0

    for m in range(99):
        slope1 = (y_gen[m-1]-y_gen[m])/0.1
        slope2 = (y_gen[m]-y_gen[m+1])/0.1
        slope_check = abs(slope1-slope2)

    if slope_check <= 1E-3:
        solution_found_ = 1

    return solution_found_, slope_check

# def main():

# STEP 4: Instantiate generator and discriminator classes
# These numbers are set up for nFeatures = 100
input_dimG = 20
modelG = Generator(input_dimG, 32, 32, 64, 64, n_traj_pts)
modelD = Discriminator(n_traj_pts, 64, 32, 16, 1)
modelG.to(device)
modelD.to(device)

# STEP 5: Instantiate loss and optimizer
criterion = nn.SmoothL1Loss().to(device)  # nn.BCEWithLogitsLoss()  # nn.CrossEntropyLoss()
loss_dg = nn.BCELoss().to(device)

learning_rate_gen = 0.01
learning_rate_disc = 0.01
optimizerD = torch.optim.Adam(modelD.parameters(), lr=learning_rate_disc)
optimizerG = torch.optim.SGD(modelG.parameters(), lr=learning_rate_gen)

# In a linear layer y = A x + B; our parameters A and B
# Hence each linear layer has 2 groups of parameters A and B
# First linear layer size ell_*2 here 2 is input size and ell_
# is next layers size (here the output)
# Readout layer size ; the size of out output

# print('\nGenerator:')
# print('\t length of model', len(list(theDecoder.parameters())))
# print('\t first layer parameter', list(theDecoder.parameters())[0].size())
# print('\t first layer bias', list(theDecoder.parameters())[1].size())
# print('\t second layer parameter', list(theDecoder.parameters())[2].size())
# print('\t second layer bias', list(theDecoder.parameters())[3].size())
#
# print('\nDiscriminator: ')
# print('\t length of model', len(list(theDiscriminator.parameters())))
# print('\t first layer parameter', list(theDiscriminator.parameters())[0].size())
# print('\t first layer bias', list(theDiscriminator.parameters())[1].size())
# print('\t second layer parameter', list(theDiscriminator.parameters())[2].size())
# print('\t second layer bias', list(theDiscriminator.parameters())[3].size())

# STEP 7: Train the GAN

# Label for fake and real data"""
def ones_target(size):
    #   Tensor containing ones, with shape = size    """
    data = Variable(torch.ones(size, 1)).to(device)
    return data


def zeros_target(size):
    #   Tensor containing zeros, with shape = size    """
    data = Variable(torch.zeros(size, 1)).to(device)
    return data


n_epochs = 100
n_iter = 0
w1 = 0.9  # 0.99  # Weight for D1 (NN Discriminator)
w2 = 0.1  # 0.05  # 0.01  # Weight for D2 (Set of rules Discriminator)
error_of_d2 = 0
G_losses = []
D_losses = []
D_losses_test = []
D_out_test = []
rules_out = []
for epoch in range(n_epochs):
    print('\n======== Epoch ', epoch, ' ========')
    for i, (traj) in enumerate(train_loader):
        # DISCRIMINATOR
        # Reset gradients
        optimizerD.zero_grad()
        modelD.zero_grad()

        # Train with real examples from dataset
        # print(traj)
        inputs = traj.float()  # .to(device)
        # zReal = np.array(zReal, dtype=np.float32)

        y_d = modelD(inputs)

        # Calculate error and backpropagate
        error_real = loss_dg(y_d, ones_target(len(traj)))

        #error_real.backward()
        #optimizerD.step()

        # # Train with fake examples from generator
        z = torch.randn([my_batch_size, input_dimG])  # torch.distributions.uniform.Uniform(-10, 10).sample([myBatchSize, dimGenInput]).to(device)  # latent vector
        y_g = modelG(z)
        y_d_fake = modelD(y_g)
        error_fake = loss_dg(y_d_fake, zeros_target(my_batch_size))

        #z2 = torch.distributions.uniform.Uniform(-10, 10).sample([myBatchSize, dimGenInput]).to(device)  # latent vector
        #y_g2 = theDecoder(z2)
        #y_d_fake2 = theDiscriminator(y_g2)
        solution_found, slope_check = setofrules(inputs.reshape([100, 1]))
        torch.tensor(solution_found, dtype=torch.int)
        if solution_found == 1:
            d2 = 0
            # rulesScore = slope_check.detach()
        else:
             d2 = 1
             error_of_d2 = slope_check.detach()  # 0.1
        # Calculate error and backpropagate
        #error_fake2 = loss_decoder(y_d_fake2, zeros_target(myBatchSize))
        #err = torch.tensor([error_fake, error_fake2])
        # error_fake2.backward()
        # optimizerD.step()
        # Compute error of D as sum over the fake and the real batches"""
        errorD = error_fake + error_real
        #rulesScore.backward()
        errorD.backward()

        # Update D
        optimizerD.step()

        # # GENERATOR
        # # Reset gradients
        # theDecoder.zero_grad()
        # optimizerG.zero_grad()
        #
        # z = torch.distributions.uniform.Uniform(-1, 1).sample([myBatchSize, dimGenInput]).to(device)  # latent vector
        # yGenerator = theDecoder(z)
        # y_d_fakeG = theDiscriminator(yGenerator)  # Since we just updated D, perform another forward pass of all-fake batch through D
        #
        # # Using D2
        # # print(yGenerator.size())

        #
        # # Calculate error and backpropagate """
        # errorG = loss_decoder(y_d_fakeG, ones_target(myBatchSize))
        # error_gd1d2 = (w1 * errorG) # + (w2 * rulesScore)
        # error_gd1d2.backward()
        # # # # Update G
        # optimizerG.step()
        #
        # loss = criterion(zReal, yGenerator)  # Difference between real and fake trajectory points

        n_iter += 1
        # Save Losses for plotting later
        # lossRecordG.append(errorG.item())
        D_losses.append(errorD.item())
        rules_out.append(solution_found)
        #D_losses.append(rulesScore)

        # if nIter == (len(train_loader.dataset) / nFeatures) * nEpochs:  # i % 5 == 0:
        #     print('Iteration: {}. Loss: {}'.format(nIter, abs(loss.item())))
        #     Disc_percentage = torch.mean(y_d_fakeG)
        #     print('Discriminator output: this is {} % real'.format(100 - (Disc_percentage.item() * 100)))

# Specify a path
# PATH = "trained_disc.pt"

# Save
# torch.save(theDiscriminator, PATH)
torch.save(modelD.state_dict(), 'discriminator_model.pt')


for i, (test_traj) in enumerate(test_loader):
    # Test with real examples from dataset
    # print(traj)
    inputs_test = test_traj.float()  # .to(device)
    # zReal = np.array(zReal, dtype=np.float32)

    y_d_test = modelD(inputs_test)

    # Calculate error and backpropagate
    #error_test = loss_decoder(y_d_test, ones_target(len(test_traj)))
    #D_losses_test.append(error_test.item())
    D_out_test.append(y_d_test.item())
stop_clock = time.time()
elapsed_hours = int((stop_clock - start_clock)//3600)
elapsed_minutes = int((stop_clock - start_clock)//60 - elapsed_hours*60)
elapsed_seconds = (stop_clock - start_clock) - elapsed_hours*3600 - elapsed_minutes*60

print('\nElapsed time ' + str(elapsed_hours) + ':' + str(elapsed_minutes) + ':' + "%.2f" % elapsed_seconds)

# # STEP 8: Plot a few sample generator results
# t = torch.linspace(0, 10, nFeatures).to("cpu")
# fig, ax = plt.subplots(5, 2)
#
# px = 1/plt.rcParams['figure.dpi']
# fig.set_figwidth(1200*px)
# fig.set_figheight(1100*px)
# for m1 in range(0, 5):
#     for m2 in range(0, 2):
#         z = torch.distributions.uniform.Uniform(-1, 1).sample([1, dimGenInput]).to(device)
#         y = theDecoder(z).to("cpu")
#
#         y_plt = y[0].detach().numpy()
#         ax[m1, m2].plot(t, y_plt)
        # y_plt_actual = zReal[0].detach().numpy()
        # ax[m1, m2].plot(t, y_plt_actual)
        # plt.legend(["Generated", "Actual"])


# figName = "Results/2023-01-14-LTI/lti_1d_ep" + str(nEpochs) + "_ex" + \
#            str(len(train_loader.dataset) // nFeatures) + "_pt" + str(nFeatures) + ".png"
# figName = "Results/2023-01-14-LTI/lti_2d_ep" + str(nEpochs) + "_ex" + \
#            str(len(train_loader.dataset)) + "_pt" + str(nFeatures) + ".png"


# fig.savefig(figName, bbox_inches='tight')
#

# fig2 = plt.figure()
# fig2.set_figwidth(1200 * px)
# fig2.set_figheight(800 * px)

fig1 = plt.plot(D_losses, label='train')
plt.title("Discriminator loss training")
plt.xlabel("iteration")
plt.ylabel("D loss")
plt.show()
plt.plot(G_losses, label="G")

fig2 = plt.plot(D_losses_test, label='D_loss')
plt.plot(D_out_test, label='D_output')
plt.title("Discriminator test data")
plt.xlabel("No. of test examples")
plt.ylabel("D output")
plt.legend()
plt.show()

x_df = pd.DataFrame(rules_out)
# x_df.to_csv('hist_data.csv')

plt.hist(x_df, bins=10)
plt.show()

