# Script for testing a generator model.
# Takes several sample G outputs and compares against a library
# using L2 distance as a similarity test.
# Also checks for diversity of samples, i.e., whether the distance between samples is zeros.


# ========== Dependencies
import torch
import numpy as np
import model_setup
import matplotlib.pyplot as plt


# =========== L2 distance calculator
def ell2_distance(z1_, z2_):
    # z1 and z2 are two vectors of equal size
    distance_ = np.sum(np.square(z1_ - z2_))
    return distance_


# ===== Load generator model
modelG = model_setup.Generator()
modelG.load_state_dict(torch.load('simple_dynamics_generator.pt'))


# ===== Generate several sample outputs
nGSample = 10
zLatent = torch.randn([nGSample, model_setup.dimGenInput])
yGen = modelG(zLatent)


# ===== Calculate distances among samples (diversity)
for m1 in range(0, nGSample):
    for m2 in range(m1, nGSample):
        ell2_distance(yGen[m1].detach().numpy(), yGen[m2].detach().numpy())


# ===== Plot a few sample G outputs
# npRow = 5
# npCol = 2
# timePlot = torch.linspace(0, 10, model_setup.nTimeStamps).to("cpu")
# px = 1/plt.rcParams['figure.dpi']
# fig, ax = plt.subplots(npRow, npCol)
# fig.set_figwidth(1800*px)
# fig.set_figheight(1500*px)
# theDecoder.to("cpu")
# for m1 in range(0, npRow):
#     for m2 in range(0, npCol):
#         x1Plot = yGen[m1 * npCol + m2][0:model_setup.nTimeStamps].detach().numpy()
#         x2Plot = yGen[m1 * npCol + m2][model_setup.nTimeStamps:model_setup.nFeatures].detach().numpy()
#         ax[m1, m2].plot(timePlot, x1Plot)
#         ax[m1, m2].plot(timePlot, x2Plot)
#
# plt.show()


# ===== Load examples from test data
nTestData = 10
testDataset = np.genfromtxt('Data/lti1d_uncertainA_all.csv', delimiter=',')
print(np.size(testDataset, 0))
print(np.size(testDataset, 1))

print(testDataset[0])

# ===== Calculate sample distances to test data (similarity)
# for m1 in range(0, nGSample):
#     for m2 in range(0, nTestData):
#         ell2_distance(yGen[m1].detach().numpy(), yTest)