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


def visualize_mu_reconstructions(model, dataloader, device, num_images=6):
    model.eval()

    images = next(iter(dataloader))
    images = images[:num_images].to(device)

    with torch.no_grad():
        mu, logvar = model.encoder(images)
        reconstructed = model.decoder(mu)

    original = images.cpu().permute(0, 2, 3, 1).numpy()
    reconstructed = reconstructed.cpu().permute(0, 2, 3, 1).numpy()

    fig, axes = plt.subplots(2, num_images, figsize=(15, 5))

    for i in range(num_images):
        axes[0, i].imshow(original[i])
        axes[0, i].axis("off")
        axes[1, i].imshow(reconstructed[i])
        axes[1, i].axis("off")

    axes[0, 0].set_title("Original")
    axes[1, 0].set_title("μ Reconstruction")

    plt.tight_layout()
    plt.show()


def get_beta(epoch, total_epochs):
    # progress = epoch / (total_epochs - 1)
    # beta = max_beta * progress
    beta = MAX_BETA * epoch / (total_epochs - 1)
    return beta


def inspect_latent_statistics(model, dataloader, device):

    model.eval()

    all_mu = []
    all_logvar = []

    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            mu, logvar = model.encoder(images)
            all_mu.append(mu.cpu())
            all_logvar.append(logvar.cpu())

    all_mu = torch.cat(all_mu, dim=0)
    all_logvar = torch.cat(all_logvar, dim=0)

    print("\n--- MU STATISTICS ---")
    print("Shape:", all_mu.shape)
    print("Mean:", all_mu.mean().item())
    print("Std:", all_mu.std().item())
    print("Min:", all_mu.min().item())
    print("Max:", all_mu.max().item())

    print("\n--- LOGVAR STATISTICS ---")
    print("Mean:", all_logvar.mean().item())
    print("Std:", all_logvar.std().item())
    print("Min:", all_logvar.min().item())
    print("Max:", all_logvar.max().item())

    variance = torch.exp(all_logvar)

    print("\n--- VARIANCE STATISTICS ---")
    print("Mean:", variance.mean().item())
    print("Std:", variance.std().item())
    print("Min:", variance.min().item())
    print("Max:", variance.max().item())


def visualize_latent_dimensions(model, dataloader, device):
    model.eval()

    all_mu = []

    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            mu, _ = model.encoder(images)
            all_mu.append(mu.cpu())

    all_mu = torch.cat(all_mu, dim=0)
    plt.figure(figsize=(8, 6))
    plt.scatter(all_mu[:, 0], all_mu[:, 1])

    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.title("Latent Space: μ₁ vs μ₂")

    plt.show()


def load_last_best_state():
    checkpoint = torch.load("vae_best.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Loaded best model from epoch:", checkpoint["epoch"])
    print("Best validation loss:", checkpoint["val_loss"])

    # visualization and inspection
    # visualize_reconstructions(model, val_loader, device)
    # inspect_latent_statistics(model, val_loader, device)


def inspect_latent_dimensions(model, dataloader, device):
    model.eval()

    all_mu = []
    all_logvar = []

    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            mu, logvar = model.encoder(images)
            all_mu.append(mu.cpu())
            all_logvar.append(logvar.cpu())

    all_mu = torch.cat(all_mu, dim=0)
    all_logvar = torch.cat(all_logvar, dim=0)

    mu_std_per_dimension = all_mu.std(dim=0)
    logvar_std_per_dimension = all_logvar.std(dim=0)

    print("\n--- PER-DIMENSION MU STD ---")

    for i, value in enumerate(mu_std_per_dimension):
        print(f"Latent {i:3d}: {value.item():.8f}")

    print("\n--- PER-DIMENSION LOGVAR STD ---")

    for i, value in enumerate(logvar_std_per_dimension):
        print(f"Latent {i:3d}: {value.item():.8f}")


def plot_latent_dimension_usage(model, dataloader, device):
    model.eval()
    all_mu = []

    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            mu, _ = model.encoder(images)
            all_mu.append(mu.cpu())

    all_mu = torch.cat(all_mu, dim=0)
    mu_std = all_mu.std(dim=0)
    plt.figure(figsize=(12, 5))

    plt.bar(range(len(mu_std)), mu_std.numpy())

    plt.xlabel("Latent Dimension")
    plt.ylabel("Standard Deviation Across Images")
    plt.title("Latent Dimension Usage")

    plt.tight_layout()
    plt.show()


def compare_latent_effects(model, dataloader, device, num_images=4):
    model.eval()

    images = next(iter(dataloader))
    images = images[:num_images].to(device)

    with torch.no_grad():
        mu, logvar = model.encoder(images)

        # Normal stochastic latent
        z_normal = reparameterize(mu, logvar)

        # Deterministic latent
        z_mu = mu

        # Completely zero latent
        z_zero = torch.zeros_like(mu)

        # Random latent from standard normal
        z_random = torch.randn_like(mu)

        # Decode all versions
        recon_normal = model.decoder(z_normal)
        recon_mu = model.decoder(z_mu)
        recon_zero = model.decoder(z_zero)
        recon_random = model.decoder(z_random)

    fig, axes = plt.subplots(5, num_images, figsize=(15, 12))

    for i in range(num_images):

        axes[0, i].imshow(images[i].cpu().permute(1, 2, 0).numpy())
        axes[0, i].axis("off")

        axes[1, i].imshow(recon_normal[i].cpu().permute(1, 2, 0).numpy())
        axes[1, i].axis("off")

        axes[2, i].imshow(recon_mu[i].cpu().permute(1, 2, 0).numpy())
        axes[2, i].axis("off")

        axes[3, i].imshow(recon_zero[i].cpu().permute(1, 2, 0).numpy())
        axes[3, i].axis("off")

        axes[4, i].imshow(recon_random[i].cpu().permute(1, 2, 0).numpy())
        axes[4, i].axis("off")

    axes[0, 0].set_title("Original")
    axes[1, 0].set_title("Normal z")
    axes[2, 0].set_title("μ")
    axes[3, 0].set_title("Zero z")
    axes[4, 0].set_title("Random z")

    plt.tight_layout()
    plt.show()


def analyze_latent_dimensions(model, dataloader, device):
    model.eval()

    all_mu = []
    all_logvar = []

    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)

            mu, logvar = model.encoder(images)

            all_mu.append(mu.cpu())
            all_logvar.append(logvar.cpu())

    all_mu = torch.cat(all_mu, dim=0)
    all_logvar = torch.cat(all_logvar, dim=0)

    variance = torch.exp(all_logvar)

    kl_per_dimension = 0.5 * (all_mu.pow(2) + variance - 1 - all_logvar)

    mean_kl_per_dimension = kl_per_dimension.mean(dim=0)

    print("\n--- PER-DIMENSION KL ---")

    for i, value in enumerate(mean_kl_per_dimension):
        print(f"Latent {i:3d}: KL = {value.item():.8f}")

    print("\n--- SUMMARY ---")

    print("Active dimensions (> 0.000):", (mean_kl_per_dimension > 0.000).sum().item()) 

    print(
        "Active dimensions (> 0.00005):", (mean_kl_per_dimension > 0.00005).sum().item()
    )

    print(
        "Active dimensions (> 0.0001):", (mean_kl_per_dimension > 0.0001).sum().item()
    )

    print(
        "Active dimensions (> 0.00025):", (mean_kl_per_dimension > 0.00025).sum().item()
    )

    print(
        "Active dimensions (> 0.0005):", (mean_kl_per_dimension > 0.0005).sum().item()
    )

    print(
        "Active dimensions (> 0.00075):", (mean_kl_per_dimension > 0.00075).sum().item()
    )

    print(
        "Active dimensions (> 0.001):", (mean_kl_per_dimension > 0.001).sum().item()
    )

    print(
        "Active dimensions (> 0.005):", (mean_kl_per_dimension > 0.005).sum().item()
    )

    print("Maximum dimension KL:", mean_kl_per_dimension.max().item())

    plt.figure(figsize=(10, 5))

    plt.bar(range(len(mean_kl_per_dimension)), mean_kl_per_dimension.numpy())

    plt.xlabel("Latent Dimension")
    plt.ylabel("Mean KL")
    plt.title("KL Contribution per Latent Dimension")

    plt.show()


if __name__ == "__main__":
    SEED = 42

    random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

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
    """
        --- OLD (experimental) ---
        β = 0
        β = .001
        β = .005
        β = .01
        β = .02
        β = .05
        β = .1

        --- NEW (zooming in on interesting parts) ---
        β = 0
        β = 0.00005
        β = 0.0001
        β = 0.00025
        β = 0.0005
        β = 0.00075
        β = 0.001

        --- NEXT BETA VALUE ZOOM IN SEQUENCE
        β = 0
        β = 0.00001
        β = 0.000025
        β = 0.00005
        β = 0.000075
        β = 0.0001
        β = 0.000125
        β = 0.00015
        β = 0.000175
        β = 0.0002
        β = 0.00025
    """
    # MAX_BETA = 0.0
    # MAX_BETA = 1.0
    # MAX_BETA = 0.1
    # MAX_BETA = 0.001
    # MAX_BETA = 0.005

    # zooming in on interesting part of the curve (from notes.md, between 0.0 & 0.001)
    # MAX_BETA = 0.00005
    # MAX_BETA = 0.0001
    # MAX_BETA = 0.00025
    # MAX_BETA = 0.0005
    # MAX_BETA = 0.00075  
    # MAX_BETA = 0.001
    # MAX_BETA = 0.005

    # next zooming in values to locate sweet spot for beta
    MAX_BETA = 0.00001



    train_total_history = []
    train_recon_history = []
    train_kl_history = []

    val_total_history = []
    val_recon_history = []
    val_kl_history = []

    best_val_loss = float("inf")

    print("USING MAX BETA OF", MAX_BETA)

    for epoch in range(EPOCHS):

        beta = get_beta(epoch, EPOCHS)

        # ------------------
        # Training
        # ------------------
        train_loss, train_recon, train_kl = train_one_epoch(
            model, train_loader, optimizer, device, beta=beta
        )

        # -------------------
        # Validation
        # -------------------
        val_loss, val_recon, val_kl = validate(model, val_loader, device, beta=beta)

        # -------------------
        # Save history
        # -------------------
        train_total_history.append(train_loss)
        train_recon_history.append(train_recon)
        train_kl_history.append(train_kl)

        val_total_history.append(val_loss)
        val_recon_history.append(val_recon)
        val_kl_history.append(val_kl)

        # -------------------
        # Save best model
        # ------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "latent_dim": 128,
                "max_beta": MAX_BETA,
                "epoch": epoch + 1,
                "val_loss": val_loss,
            }
            torch.save(checkpoint, "vae_best.pth")
            print("Saved new best model.")

        # ---------------------
        # Print statistics
        # ---------------------
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"| Beta: {beta:.6f} "
            f"| Train Loss: {train_loss:.6f} "
            f"| Train Recon: {train_recon:.6f} "
            f"| Train KL: {train_kl:.8f} "
            f"| Val Loss: {val_loss:.6f} "
            f"| Val Recon: {val_recon:.6f} "
            f"| Val KL: {val_kl:.6f}"
        )

    # ---------------------
    # very important, load the best model
    # --------------------
    load_last_best_state()

    # plot histories
    plot_training_history(train_total_history, val_total_history, "Total VAE Loss")
    plot_training_history(train_recon_history, val_recon_history, "Reconstruction Loss")
    plot_training_history(train_kl_history, val_kl_history, "KL Divergence")

    # visualize reconstruction after passing in an image
    visualize_reconstructions(model, val_loader, device)

    # inspect our latent space
    inspect_latent_statistics(model, val_loader, device)

    # visualize only mu reconstruction
    visualize_mu_reconstructions(model, val_loader, device)

    # visualize stripped down latent dimension
    visualize_latent_dimensions(model, val_loader, device)

    # inspect indepth latent dimension
    inspect_latent_dimensions(model, val_loader, device)

    # plot bar chart latent dimension usage
    plot_latent_dimension_usage(model, val_loader, device)

    # compare the latent space effects
    compare_latent_effects(model, val_loader, device)

    # analyze specific/individual latent dimension numbers
    analyze_latent_dimensions(model, val_loader, device)
