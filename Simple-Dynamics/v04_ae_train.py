# ========== Dependencies
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
# from datetime import datetime
import model_setup


# ===== Initialize
device = model_setup.device
start_clock = time.time()

nFeatures = model_setup.nFeatures
myBatchSize = model_setup.myBatchSize

# ===== Instantiate encoder, loss function, and optimizer
theEncoder = model_setup.Encoder()
theEncoder.to(device)
loss_encoder = nn.MSELoss().to(device)
learningRateE = 1E-4
optimizerE = torch.optim.Adam(theEncoder.parameters(), lr=learningRateE)

# ===== Instantiate decoder (generator), loss function, and optimizer
theDecoder = model_setup.DecoderGenerator()
theDecoder.to(device)
loss_decoder = nn.MSELoss().to(device)
learningRateG = 1E-4
optimizerG = torch.optim.Adam(theDecoder.parameters(), lr=learningRateG)

# ===== Initialize training
nEpochs = 3000
nIter = 0
lossRecordEn = []

# ===== Train autoencoder
for epoch in range(nEpochs):
    print('\n======== Epoch ', epoch, ' ========')
    for i, (traj) in enumerate(model_setup.trainLoader):

        theDecoder.zero_grad()
        theEncoder.zero_grad()

        # input real examples from dataset
        zReal = traj.float().to(device)

        # ----- Pass real data through encoder
        yEncoder = theEncoder(zReal)

        # ----- Pass encoder output through generator (generator acts as decoder)
        yGenerator = theDecoder(yEncoder).to(device)

        # Difference between generated and real data
        aeError = zReal - yGenerator

        # ----- Find gradient and update D
        aeLoss = loss_encoder(aeError, model_setup.zeros_target(nFeatures))
        aeLoss.backward()
        optimizerE.step()
        optimizerG.step()

        # ----- Book-keeping
        nIter += 1

        # Save Losses for plotting later
        lossRecordEn.append(aeLoss.item())

# ===== Save the trained autoencoder model (Generator which is the decoder)
torch.save(theDecoder.state_dict(), 'ltiDecoder.pt')

stopClock = time.time()
elapsedHours = int((stopClock - start_clock) // 3600)
elapsedMinutes = int((stopClock - start_clock) // 60 - elapsedHours * 60)
elapsedSeconds = (stopClock - start_clock) - elapsedHours * 3600 - elapsedMinutes * 60

print('\nElapsed time ' + str(elapsedHours) + ':' + str(elapsedMinutes) + ':' + "%.2f" % elapsedSeconds)

# Plot loss
fig1 = plt.figure()
plt.plot(lossRecordEn)
plt.title("Loss during autoencoder training")
plt.xlabel("Iteration")
plt.ylabel("Loss")


# Plot a few sample generator results
nPlotRow = 5
nPlotCol = 2
# fig3 = plt.figure()
fig, ax = plt.subplots(nPlotRow, nPlotCol)

# --- For 2 state
t = torch.linspace(0, 10, model_setup.nTimeStamps).to("cpu")
px = 1/plt.rcParams['figure.dpi']
fig.set_figwidth(1800*px)
fig.set_figheight(1500*px)
theDecoder.to("cpu")
for m1 in range(0, nPlotRow):
    for m2 in range(0, nPlotCol):
        z = torch.randn([1, model_setup.dimLatent])
        y = theDecoder(z)
        y = y.reshape([nFeatures, 1])
        x1Plot = y[0:model_setup.nTimeStamps].detach().numpy()
        # x1Plot = zReal[0][0:model_setup_G_rulesD.nTimeStamps].detach().numpy()
        x2Plot = y[model_setup.nTimeStamps:nFeatures].detach().numpy()
        ax[m1, m2].plot(t, x1Plot)
        ax[m1, m2].plot(t, x2Plot)

plt.show()
