import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data into a DataFrame
data = pd.read_csv("data_linear_n2m1_wUnmodeled_woNoise.csv").to_numpy()

# Remove first row of data
data = data[1:, :]

# Number of pressure sensors
n = 32

# Number of timesteps
N = len(data)

# Bucket size
b = 25

# Jump length
J = 1

# Input and output data
out = data[:, 1:3]
in_data = data[:, 3:]

# Z-Score standardize the heave and pitch data
scalar = StandardScaler()
scalar.fit(out)
out = scalar.transform(out)

# Scale outputs to be the same order of magnitude
# out[:,0] *= 10
# out[:,1] *= 10000

outStore = []
inStore = []
bin = [0, b]
for k in range(int(np.ceil((N - b) / J))):
    idx = np.arange(bin[0], bin[1])
    outStore.append(out[bin[1], :].reshape(1, -1).T)
    inStore.append(in_data[idx, :].reshape(-1, 1))
    bin = [bin[0] + J, bin[1] + J]

# Convert lists to single NumPy arrays
ydata = np.array(outStore, dtype=np.float32).squeeze()
xdata = np.array(inStore, dtype=np.float32).squeeze()

# Randomly generate testing and training data
x_train, x_test, y_train, y_test = train_test_split(xdata, ydata, test_size=0.20, shuffle=True)

# Convert NumPy arrays to PyTorch tensors
x_train = torch.tensor(x_train)
x_test = torch.tensor(x_test)
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)

# Print dimensions of tensors for user
print(x_train.shape)
print(y_train.shape)

# Save tensors to file
torch.save(x_train,"xTrainTensor.pt")
torch.save(y_train,"yTrainTensor.pt")
torch.save(x_test,"xTestTensor.pt")
torch.save(y_test,"yTestTensor.pt")
