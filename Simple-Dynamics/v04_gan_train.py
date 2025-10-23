# ========== Dependencies
import torch
import torch.nn as nn
from torch.autograd.variable import Variable
import matplotlib.pyplot as plt
import time
import model_setup


# ===== Initialize
device = model_setup.device
start_clock = time.time()
nFeatures = model_setup.nFeatures
myBatchSize = model_setup.myBatchSize

# ===== Load autoencoder as generator, and instantiate loss function
theGenerator = model_setup.DecoderGenerator()
theGenerator.load_state_dict(torch.load('ltiDecoder.pt'))
theGenerator.to(device)
loss_gen_rules = nn.MSELoss().to(device)
learningRateG = 1E-4
optimizerG = torch.optim.Adam(theGenerator.parameters(), lr=learningRateG)


# ===== Instantiate discriminator, loss function, and optimizer
theDiscriminator = model_setup.Discriminator()
theDiscriminator.to(device)
loss_disc = nn.BCELoss().to(device)
learningRateD = 1E-4
optimizerD = torch.optim.Adam(theDiscriminator.parameters(), lr=learningRateD)


# ===== Initialize training
nEpochs = 5000
nIter = 0
lossRecordD = []
lossRecordG = []


# ===== Train
for epoch in range(nEpochs):
    print('\n======== Epoch ', epoch, ' ========')
    for i, (traj) in enumerate(model_setup.trainLoader):

        # ===== DISCRIMINATOR TRAINING =====
        # Reset gradients
        theDiscriminator.zero_grad()

        # ----- Real examples from dataset
        zReal = traj.float().to(device)
        thisBatchSize = len(zReal)

        # ----- Calculate loss in nn D with real data
        yDiscReal = theDiscriminator(zReal).to(device)
        lossDReal = loss_disc(yDiscReal, model_setup.ones_target(thisBatchSize))

        # ----- Calculate loss in nn D with fake data
        nLatent = torch.randn([myBatchSize, model_setup.dimLatent]).to(device)
        zFake = theGenerator(nLatent).to(device)
        yDiscFake = theDiscriminator(zFake).to(device)
        lossDFake = loss_disc(torch.transpose(yDiscFake, 0, 1), model_setup.zeros_target(myBatchSize))

        # ----- Find gradient and update D
        lossD = lossDReal + lossDFake
        lossD.backward()
        optimizerD.step()

        if (epoch < 500) & (epoch % 5):
            continue

        # ===== GENERATOR TRAINING =====
        # ----- Reset gradients
        theGenerator.zero_grad()

        # ----- Calculate loss in G from nn D
        nLatent = torch.randn([myBatchSize, model_setup.dimLatent]).to(device)
        zGenerated = theGenerator(nLatent).to(device)

        yDiscGenTrain = theDiscriminator(zGenerated)
        lossGDisc = loss_disc(yDiscGenTrain, model_setup.ones_target(myBatchSize))

        rulesScore = model_setup.rules_d_state1(zGenerated, device)
        lossGRules = loss_gen_rules(rulesScore, Variable(torch.zeros(myBatchSize, nFeatures)).to(device))

        # ----- Find gradient and update G
        lossG = lossGDisc + 0.05*lossGRules
        lossG.backward()
        optimizerG.step()

        # ----- Book-keeping
        nIter += 1

        # Save Losses for plotting later
        lossRecordD.append(lossD.item())
        lossRecordG.append(lossG.item())


# # ===== Save the trained model
# torch.save(model_D.state_dict(), 'simple_dynamics_discriminator_nn.pt')

stopClock = time.time()
elapsedHours = int((stopClock - start_clock) // 3600)
elapsedMinutes = int((stopClock - start_clock) // 60 - elapsedHours * 60)
elapsedSeconds = (stopClock - start_clock) - elapsedHours * 3600 - elapsedMinutes * 60

print('\nElapsed time ' + str(elapsedHours) + ':' + str(elapsedMinutes) + ':' + "%.2f" % elapsedSeconds)

# Plot discriminator loss
fig1 = plt.figure()
plt.plot(lossRecordD)
plt.title("Discriminator loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")

# Plot generator loss
fig2 = plt.figure()
plt.plot(lossRecordG)
plt.title("Generator loss during training")
plt.xlabel("Iteration")
plt.ylabel("Generator Loss")


# Plot a few sample generator results
nPlotRow = 5
nPlotCol = 2
fig3 = plt.figure()
fig, ax = plt.subplots(nPlotRow, nPlotCol)

# --- For 2 state
t = torch.linspace(0, 10, model_setup.nTimeStamps).to("cpu")
px = 1/plt.rcParams['figure.dpi']
fig.set_figwidth(1800*px)
fig.set_figheight(1500*px)
theGenerator.to("cpu")
for m1 in range(0, nPlotRow):
    for m2 in range(0, nPlotCol):
        nLatent = torch.randn([1, model_setup.dimLatent])
        y = theGenerator(nLatent)
        y = y.reshape([nFeatures, 1])
        x1_plt = y[0:model_setup.nTimeStamps].detach().numpy()
        x2_plt = y[model_setup.nTimeStamps:nFeatures].detach().numpy()
        ax[m1, m2].plot(t, x1_plt)
        ax[m1, m2].plot(t, x2_plt)

plt.show()




# FIX BATCH SIZE ISSUE
# WRITE SCRIPTS FOR GENERATOR PERFORMANCE INDICES