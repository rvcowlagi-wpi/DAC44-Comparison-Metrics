import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
# Hyperparameters
hidden_size_E1, hidden_size_E2, hidden_size_E3, hidden_size_E4, hidden_size_E5 = 225, 196, 125, 100, 81
hidden_size_D1, hidden_size_D2, hidden_size_D3, hidden_size_D4, hidden_size_D5 = 81, 100, 125, 196, 225
latent_size = 32
learning_rate = 1e-3
epochs = 1000
n_features = 2057
batch_size = 32


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_size_E1),
            nn.ReLU(),
            nn.Linear(hidden_size_E1, hidden_size_E2),
            nn.ReLU(),
            nn.Linear(hidden_size_E2, hidden_size_E3),
            nn.ReLU(),
            nn.Linear(hidden_size_E3, hidden_size_E4),
            nn.ReLU(),
            nn.Linear(hidden_size_E4, hidden_size_E5),
            nn.ReLU(),
            nn.Linear(hidden_size_E5, latent_size * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size_D1),
            nn.ReLU(),
            nn.Linear(hidden_size_D1, hidden_size_D2),
            nn.ReLU(),
            nn.Linear(hidden_size_D2, hidden_size_D3),
            nn.ReLU(),
            nn.Linear(hidden_size_D3, hidden_size_D4),
            nn.ReLU(),
            nn.Linear(hidden_size_D4, hidden_size_D5),
            nn.ReLU(),
            nn.Linear(hidden_size_D5, n_features)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        latent_params = self.encoder(x)
        mu, logvar = torch.chunk(latent_params, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar


def train_vae(vae, train_loader, optimizer, loss_mse):
    vae.train()
    best_loss = float('inf')
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in train_loader:
            inputs = batch[0].float().to(device)  # Extract the data tensor
            x_recon, mu, logvar = vae(inputs)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss_recon = loss_mse(x_recon, inputs)
            loss = kl_loss + loss_recon
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(vae.state_dict(), "best_zvae_model.pth")


def generate_samples(vae, num_samples=1000, output_file="generated_samples.csv"):
    vae.eval()
    samples = []
    with torch.no_grad():
        for _ in range(num_samples):
            z = torch.randn(1, latent_size).to(device)
            generated = vae.decoder(z).cpu().numpy()
            samples.append(generated)

    samples = np.vstack(samples)
    np.savetxt(output_file, samples, delimiter=",")
    print(f"Generated {num_samples} samples saved to '{output_file}'.")


def main():
    vae = VAE().to(device)
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
    loss_mse = nn.MSELoss(reduction="sum").to(device)

    # Load and prepare data
    data_minthreat = np.array(pd.read_csv("Data/data_minthreat_F175_1000_7s_company_10.txt", header=None))
    train_dataset = TensorDataset(torch.tensor(data_minthreat[:200]))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Train the model
    train_vae(vae, train_loader, optimizer, loss_mse)

    # Load the best model
    vae.load_state_dict(torch.load("best_zvae_model.pth"))

    # Generate and save 1000 samples
    generate_samples(vae, num_samples=1000, output_file="C:/Users/nubapat/OneDrive - Worcester Polytechnic Institute (wpi.edu)/2024-JNL-NUBapat/Archives/Scatter-Plots/svae_1000_10.csv")


if __name__ == "__main__":
    main()
