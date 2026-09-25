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


def show_batch(images, num_images=8):
    images = images[:num_images]
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    _, axes = plt.subplots(2, 4, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def train_one_epoch(model, dataloader, optimizer, device, beta=1.0):
    model.train()

    total_loss = 0.0
    total_reconstruction = 0.0
    total_kl = 0.0

    for images in dataloader:
        images = images.to(device)
        optimizer.zero_grad()

        # forward
        reconstructed, mu, logvar, z = model(images)

        # loss
        loss, reconstruction_loss, kl_loss = vae_loss(
            reconstructed, images, mu, logvar, beta=beta
        )

        # backward
        loss.backward()

        # step
        optimizer.step()

        # loss cleaning
        total_loss += loss.item()
        total_reconstruction += reconstruction_loss.item()
        total_kl += kl_loss.item()

    # batch cleaning
    num_batches = len(dataloader)
    return (
        total_loss / num_batches,
        total_reconstruction / num_batches,
        total_kl / num_batches,
    )


def validate(model, dataloader, device, beta=1.0):
    model.eval()
    total_loss = 0.0
    total_reconstruction = 0.0
    total_kl = 0.0

    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            reconstructed, mu, logvar, z = model(images)
            loss, reconstruction_loss, kl_loss = vae_loss(
                reconstructed, images, mu, logvar, beta=beta
            )
            total_loss += loss.item()
            total_reconstruction += reconstruction_loss.item()
            total_kl += kl_loss.item()

    num_batches = len(dataloader)
    return (
        total_loss / num_batches,
        total_reconstruction / num_batches,
        total_kl / num_batches,
    )


def plot_training_history(train_history, val_history, title):
    plt.figure(figsize=(8, 5))
    plt.plot(train_history, label="Train")
    plt.plot(val_history, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.show()


def visualize_reconstructions(model, dataloader, device, num_images=6):
    model.eval()
    images = next(iter(dataloader))
    images = images[:num_images].to(device)

    with torch.no_grad():
        reconstructed, _, _, _ = model(images)

    original = images.cpu().permute(0, 2, 3, 1).numpy()
    reconstructed = reconstructed.cpu().permute(0, 2, 3, 1).numpy()
    fig, axes = plt.subplots(2, num_images, figsize=(15, 5))

    for i in range(num_images):
        axes[0, i].imshow(original[i])
        axes[0, i].axis("off")
        axes[1, i].imshow(reconstructed[i])
        axes[1, i].axis("off")

    axes[0, 0].set_title("Original")
    axes[1, 0].set_title("Reconstruction")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    random.seed(42)

    FOLDER_PATH = "my_dataset"

    # all images
    all_image_paths = sorted(glob.glob(os.path.join(FOLDER_PATH, "*.jpg")))
    random.shuffle(all_image_paths)

    # train-test split
    split_index = int(0.8 * len(all_image_paths))
    train_paths = all_image_paths[:split_index]
    val_paths = all_image_paths[split_index:]

    print("Total images:", len(all_image_paths))
    print("Training images:", len(train_paths))
    print("Validation images:", len(val_paths))

    # train-validation dataset
    train_dataset = ImageDataset(train_paths, transform=train_transform)
    val_dataset = ImageDataset(val_paths, transform=val_transform)

    # train-validation loader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # next steps
    images = next(iter(train_loader))
    # show_batch(images)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print("Using device:", device)

    # model
    model = VAE(latent_dim=128).to(device)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    EPOCHS = 50
    BETA = 1.0

    train_total_history = []
    train_recon_history = []
    train_kl_history = []

    val_total_history = []
    val_recon_history = []
    val_kl_history = []

    # actual training loop
    for epoch in range(EPOCHS):
        train_loss, train_recon, train_kl = train_one_epoch(
            model, train_loader, optimizer, device, beta=BETA
        )

        val_loss, val_recon, val_kl = validate(model, val_loader, device, beta=BETA)

        train_total_history.append(train_loss)
        train_recon_history.append(train_recon)
        train_kl_history.append(train_kl)

        val_total_history.append(val_loss)
        val_recon_history.append(val_recon)
        val_kl_history.append(val_kl)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train Recon: {train_recon:.4f} "
            f"| Train KL: {train_kl:.4f} "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val Recon: {val_recon:.4f} "
            f"| Val KL: {val_kl:.4f}"
        )

    # plot histories
    plot_training_history(train_total_history, val_total_history, "Total VAE Loss")
    plot_training_history(train_recon_history, val_recon_history, "Reconstruction Loss")
    plot_training_history(train_kl_history, val_kl_history, "KL Divergence")

    # save model + optimizer + val loss
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "latent_dim": 128,
        "beta": BETA,
        "epoch": EPOCHS + 1,
        "val_loss": sum(val_total_history) / len(val_total_history),
    }

    best_val_loss = float("inf")
    val_loss = sum(val_total_history) / len(val_total_history)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(checkpoint, "vae_best.pth")
        print("Saved new best model.")

    # visualize reconstruction after passing in an image
    visualize_reconstructions(model, val_loader, device)
