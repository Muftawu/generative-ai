import glob
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# --- 1. Custom Dataset Loader ---
class WebcamDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.image_paths = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

# --- 2. The Autoencoder Architecture ---
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.Sigmoid() 
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

# --- 3. Visualization Function ---
def visualize_results(model, dataloader, device, num_images=5):
    """Passes a batch through the model and plots original vs reconstruction side-by-side."""
    model.eval() # Set model to evaluation mode
    
    # Grab one batch of real images
    real_images = next(iter(dataloader))
    real_images = real_images.to(device)
    
    # Run them through the model
    with torch.no_grad(): # We don't need to calculate gradients for testing
        reconstructed_images = model(real_images)
    
    # Move tensors back to CPU for plotting
    real_images = real_images.cpu().permute(0, 2, 3, 1).numpy()
    reconstructed_images = reconstructed_images.cpu().permute(0, 2, 3, 1).numpy()

    # Plotting
    fig, axes = plt.subplots(2, num_images, figsize=(15, 6))
    fig.suptitle("Top: Original Webcam Images | Bottom: AI Reconstructions", fontsize=16)
    
    for i in range(num_images):
        # Top row: Originals
        axes[0, i].imshow(real_images[i])
        axes[0, i].axis('off')
        
        # Bottom row: Reconstructions
        axes[1, i].imshow(reconstructed_images[i])
        axes[1, i].axis('off')
        
    plt.tight_layout()
    plt.show()

# --- 4. Main Execution ---
if __name__ == "__main__":
    # Settings
    MODE = "test"  # Change to "train" to resume/continue training, or "test" to see visuals
    
    FOLDER_PATH = "my_dataset"
    WEIGHTS_FILE = "autoencoder_weights.pth"
    BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor() 
    ])

    dataset = WebcamDataset(FOLDER_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize Model
    model = Autoencoder().to(device)
    
    # Check for existing weights to resume training or run inference
    if os.path.exists(WEIGHTS_FILE):
        print(f"Found existing weights at {WEIGHTS_FILE}. Loading...")
        model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=device))
    else:
        print("No existing weights found. Starting fresh.")

    if MODE == "test":
        print("Running inference and generating visual grid...")
        visualize_results(model, dataloader, device)
        
    elif MODE == "train":
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        print("Resuming training...")
        model.train() # Set model to training mode
        for epoch in range(EPOCHS):
            total_loss = 0
            
            for images in dataloader:
                images = images.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, images)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}")

        torch.save(model.state_dict(), WEIGHTS_FILE)
        print(f"Training complete! Model weights saved/updated in '{WEIGHTS_FILE}'.")   
