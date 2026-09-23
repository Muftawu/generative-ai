import glob
import os
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

train_transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ]
)


class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(64 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(64 * 8 * 8, latent_dim)

    def forward(self, x):
        features = self.conv(x)
        features = self.flatten(features)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        return mu, logvar


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    epsilon = torch.randn_like(std)
    z = mu + epsilon * std
    return z


class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 8 * 8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                16, 3, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 64, 8, 8)
        x = self.deconv(x)
        return x


class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar, z


def vae_loss(reconstructed, original, mu, logvar, beta=1.0):
    reconstruction_loss = F.mse_loss(reconstructed, original, reduction="mean")
    kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss.sum(dim=1).mean()
    total_loss = reconstruction_loss + beta * kl_loss
    return (total_loss, reconstruction_loss, kl_loss)


if __name__ == "__main__":
    model = VAE(latent_dim=128)

    x = torch.randn(4, 3, 64, 64)

    reconstructed, mu, logvar, z = model(x)

    print("Input:", x.shape)
    print("Mu:", mu.shape)
    print("Logvar:", logvar.shape)
    print("Z:", z.shape)
    print("Reconstructed:", reconstructed.shape)

    # FOLDER_PATH = "my_dataset"
    #
    # # all images
    # all_image_paths = sorted(glob.glob(os.path.join(FOLDER_PATH, "*.jpg")))
    # random.shuffle(all_image_paths)
    #
    # # train-test split
    # split_index = int(0.8 * len(all_image_paths))
    # train_paths = all_image_paths[:split_index]
    # val_paths = all_image_paths[split_index:]
    #
    # # train-validation dataset
    # train_dataset = ImageDataset(train_paths, transform=train_transform)
    # val_dataset = ImageDataset(val_paths, transform=val_transform)
    #
    # # train-validation loader
    # train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
