# ========== Dependencies
import torch
import torch.nn as nn
import numpy as np
from torch.autograd.variable import Variable
import matplotlib.pyplot as plt
import time
from datetime import datetime
import model_setup


# ===== Initialize
device = "cpu"
start_clock = time.time()

nFeatures = model_setup.nFeatures
my_batch_size = 1

# ===== Instantiate generator, loss function, and optimizer
modelG = model_setup.Generator()
modelG.to(device)
lossFcnG = nn.MSELoss().to(device)
learningRateG = 2E-4
optimizerG = torch.optim.Adam(modelG.parameters(), lr=learningRateG)

# ===== Initialize training
nEpochs = 100
nIter = 0
lossRecordG = []

# ===== Train
for epoch in range(nEpochs):
        print('\n======== Epoch ', epoch, ' ========')

        # ===== GENERATOR =====
        # ----- Reset gradients
        modelG.zero_grad()
        optimizerG.zero_grad()

        # ----- Sample from latent vector space
        z = torch.randn([my_batch_size, model_setup.dimGenInput]).to(device)

        # ----- Pass latent vector through G
        y_gG = modelG(z).to(device)

        # ----- Find rules discriminator score (ideal is zero) and loss
        rulesScore = model_setup.rules_d_state1(y_gG.reshape([nFeatures, 1]), device)
        lossGFromRules = lossFcnG(rulesScore, Variable(torch.zeros(nFeatures, 1)).to(device))

        # ----- Find gradient and update G
        lossGFromRules.backward()
        optimizerG.step()

        # ----- Book-keeping
        nIter += 1

        # ----- Save Losses for plotting later
        lossRecordG.append(lossGFromRules.item())
        print(lossGFromRules.item())

# ===== Save the trained model
torch.save(modelG.state_dict(), 'simple_dynamics_generator.pt')


stopClock = time.time()
elapsedHours = int((stopClock - start_clock) // 3600)
elapsedMinutes = int((stopClock - start_clock) // 60 - elapsedHours * 60)
elapsedSeconds = (stopClock - start_clock) - elapsedHours * 3600 - elapsedMinutes * 60

print('\nElapsed time ' + str(elapsedHours) + ':' + str(elapsedMinutes) + ':' + "%.2f" % elapsedSeconds)

# Plot a few sample generator results
n_plot_row = 5
n_plot_col = 2
# fig2 = plt.figure()
fig, ax = plt.subplots(n_plot_row, n_plot_col)

# --- For 2 states
t = torch.linspace(0, 10, model_setup.nTimeStamps).to("cpu")
px = 1/plt.rcParams['figure.dpi']
fig.set_figwidth(1800*px)
fig.set_figheight(1500*px)
modelG.to("cpu")

z = torch.randn([10, model_setup.dimGenInput])
y = modelG(z)
for m1 in range(0, n_plot_row):
    for m2 in range(0, n_plot_col):
        x1_plt = y[0][0:model_setup.nTimeStamps].detach().numpy()
        x2_plt = y[0][model_setup.nTimeStamps:nFeatures].detach().numpy()
        ax[m1, m2].plot(t, x1_plt)
        ax[m1, m2].plot(t, x2_plt)


# --- For 1 state
# t = torch.linspace(0, 10, nFeatures).to("cpu")
# px = 1/plt.rcParams['figure.dpi']
# fig.set_figwidth(900*px)
# fig.set_figheight(1500*px)
# theDecoder.to("cpu")
# for m1 in range(0, npRow):
#     for m2 in range(0, npCol):
#         z = torch.randn([myBatchSize, dimGenInput])
#         y = theDecoder(z)
#
#         x1Plot = y[0].detach().numpy()
#         ax[m1].plot(t, x1Plot)


# --- Generator loss
# fig3 = plt.figure()
# plt.plot(lossRecordG, label="G_loss")
# plt.title("Generator loss during training")
# plt.xlabel("Iteration")
# plt.ylabel("Loss")
# plt.legend()


ts = (np.datetime64(datetime.utcnow()) - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
figName = "Results/202305/rules_only_" + str(round(ts)) + ".png"
# fig.savefig(figName, bbox_inches='tight')

plt.show()