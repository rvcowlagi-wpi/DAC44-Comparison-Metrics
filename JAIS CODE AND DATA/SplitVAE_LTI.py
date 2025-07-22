#This program implements the Split-VAE for the LTI problem

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import glob
import os
# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.manual_seed(42)
def load_and_stack_csv(folder_path, file_pattern):
    # Get list of all CSV files matching the pattern
    csv_files = sorted(glob.glob(os.path.join(folder_path, file_pattern)))

    # Read and stack CSV files
    tensor_list = []

    for file in csv_files:
        df = pd.read_csv(file, header=None)
        tensor = torch.tensor(df.values, dtype=torch.float32)
        tensor = tensor.T
        tensor_list.append(tensor)
    final_tensor = torch.stack(tensor_list)
    return final_tensor
# Define folder paths and file patterns
folder_path = 'C:/Users/nubapat/OneDrive - Worcester Polytechnic Institute (wpi.edu)/2025_VRNN/case01/set09'
folder_path1 = 'C:/Users/nubapat/OneDrive - Worcester Polytechnic Institute (wpi.edu)/2025_VRNN/realcase02' # Change as per set number

# Load and stack CSV files
final_tensor_noise = load_and_stack_csv(folder_path, 'traj_*.csv')
final_tensor_noiseless = load_and_stack_csv(folder_path1, 'realtraj_*.csv')
# Reshape tensors to [1000, 100100]
final_tensor_noise = final_tensor_noise.reshape(1000, 1001 * 100)
final_tensor_noiseless = final_tensor_noiseless.reshape(1000, 1001 * 100)
final_tensor_noise = final_tensor_noise[:500,:]
final_tensor_noiseless = final_tensor_noiseless[:1000,:]
# Print final tensor shapes
print(f"Final tensor shape for case01: {final_tensor_noise.shape}")
print(f"Final tensor shape for realcase01: {final_tensor_noiseless.shape}")

# Create labels
labels_zero = torch.zeros(final_tensor_noise.shape[0], dtype=torch.long).to(device)
labels_ones = torch.ones(final_tensor_noiseless.shape[0], dtype=torch.long).to(device)
#
# Concatenate data tensors and labels
combined_data = torch.cat((final_tensor_noiseless, final_tensor_noise), dim=0)
combined_labels = torch.cat((labels_ones, labels_zero), dim=0)
combined_dataset = TensorDataset(combined_data, combined_labels)
train_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True)
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, z1_dim, z2_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.ln1 = nn.LayerNorm(hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.ln2 = nn.LayerNorm(hidden_dim2)
        self.fc21 = nn.Linear(hidden_dim2, z1_dim)
        self.fc22 = nn.Linear(hidden_dim2, z1_dim)
        self.fc31 = nn.Linear(hidden_dim2, z2_dim)
        self.fc32 = nn.Linear(hidden_dim2, z2_dim)

    def forward(self, x):
        h1 = F.relu(self.ln1(self.fc1(x)))
        h2 = F.relu(self.ln2(self.fc2(h1)))
        # h3 = F.relu(self.ln3(self.fc3(h2)))

        z1_mu = self.fc21(h2)
        z1_logvar = self.fc22(h2)
        z2_mu = self.fc31(h2)
        z2_logvar = self.fc32(h2)

        return z1_mu, z1_logvar, z2_mu, z2_logvar

class Decoder(nn.Module):
    def __init__(self, z1_dim, z2_dim, hidden_dim1, hidden_dim2, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(z1_dim + z2_dim, hidden_dim1)
        self.ln1 = nn.LayerNorm(hidden_dim1)

        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.ln2 = nn.LayerNorm(hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, z1, z2):
        x = torch.cat([z1, z2], dim=1)
        h1 = F.relu(self.ln1(self.fc1(x)))
        h2 = F.relu(self.ln2(self.fc2(h1)))
        # h3 = F.relu(self.ln3(self.fc3(h2)))
        x_ = self.fc3(h2)
        return x_

class SplitVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2,z1_dim, z2_dim):
        super(SplitVAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim1, hidden_dim2, z1_dim, z2_dim)
        self.decoder = Decoder(z1_dim, z2_dim, hidden_dim1, hidden_dim2, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        z1_mu, z1_logvar, z2_mu, z2_logvar = self.encoder(x)
        z1 = self.reparameterize(z1_mu, z1_logvar)
        z2 = self.reparameterize(z2_mu, z2_logvar)
        return self.decoder(z1, z2), z1_mu, z1_logvar, z2_mu, z2_logvar

def loss_function(recon_x, x, z1_mu, z1_logvar, z2_mu, z2_logvar, is_noiseless, lambda_reg):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    # KL divergence losses
    kl_loss_z1 = -0.5 * torch.sum(1 + z1_logvar - z1_mu.pow(2) - z1_logvar.exp())
    kl_loss_z2 = -0.5 * torch.sum(1 + z2_logvar - z2_mu.pow(2) - z2_logvar.exp())

    # Regularization term (applied only to noiseless samples)
    regularization = lambda_reg * torch.sum(z1_mu.pow(2) + z1_logvar.exp() - 1 - z1_logvar) * is_noiseless
    regularization = torch.sum(regularization)
    return recon_loss + kl_loss_z1 + kl_loss_z2 + regularization


# Sampling from latent space
def sample_from_latent_space(model, num_samples, z1_dim, z2_dim, save_path):
    # Ensure the model is in evaluation mode
    model.eval()

    # Generate random samples from the latent space
    z1_samples = torch.randn(num_samples, z1_dim).to(device)
    z2_samples = torch.randn(num_samples, z2_dim).to(device)

    # Decode the latent samples
    with torch.no_grad():
        generated_samples = model.decoder(z1_samples, z2_samples).cpu().numpy()

    # Save the generated samples to a CSV file
    np.savetxt(save_path, generated_samples, delimiter=',')
    print(f"Generated {num_samples} samples and saved to {save_path}")


def main():
    # Instantiate model and optimizer
    model = SplitVAE(input_dim=100100, hidden_dim1=625, hidden_dim2=400, z1_dim=20, z2_dim=20).to(
        device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    lambda_reg = 1
    num_epochs = 1000

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for data_batch, labels_batch in train_loader:
            data_batch = data_batch.to(device)
            labels_batch = labels_batch.to(device)
            optimizer.zero_grad()
            recon_batch, z1_mu, z1_logvar, z2_mu, z2_logvar = model(data_batch)
            loss = loss_function(recon_batch, data_batch, z1_mu, z1_logvar, z2_mu, z2_logvar, labels_batch, lambda_reg)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {train_loss / len(train_loader.dataset)}')

    # Save model
    model_path = 'splitvae_model.pth'
    torch.save(model.state_dict(), model_path)

    # Generate and save samples from the latent space
    num_samples = 1000
    z1_dim = 20
    z2_dim = 20
    save_path = 'C:/Users/nubapat/OneDrive - Worcester Polytechnic Institute (wpi.edu)/2024-JNL-NUBapat/Archives/Scatter-Plots/splitvae_set09.csv'
    sample_from_latent_space(model, num_samples, z1_dim, z2_dim, save_path)


if __name__ == "__main__":
    main()