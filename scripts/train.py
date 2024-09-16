import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np


# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.srcnn_model import SRCNN
# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, image_size=(33, 33)):
        self.lr_dir = os.path.abspath('D:/Development/image-super-resolution/data/processed_images')
        self.hr_dir = os.path.abspath('D:/Development/image-super-resolution/data/processed_images')
        self.lr_files = sorted([f for f in os.listdir(self.lr_dir) if f.endswith('.jpg')])
        self.hr_files = sorted([f for f in os.listdir(self.hr_dir) if f.endswith('.jpg')])
        self.image_size = image_size  # Fixed size for images

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, idx):
        lr_img = Image.open(os.path.join(self.lr_dir, self.lr_files[idx])).convert('RGB')
        hr_img = Image.open(os.path.join(self.hr_dir, self.hr_files[idx])).convert('RGB')

        # Resize images to a fixed size
        lr_img = lr_img.resize(self.image_size, Image.BICUBIC)
        hr_img = hr_img.resize(self.image_size, Image.BICUBIC)
        
        lr_img = np.array(lr_img).transpose((2, 0, 1)) / 255.0
        hr_img = np.array(hr_img).transpose((2, 0, 1)) / 255.0
        
        return torch.tensor(lr_img, dtype=torch.float32), torch.tensor(hr_img, dtype=torch.float32)

def train_model(lr_dir, hr_dir, epochs=10):
    model = SRCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    dataset = SRDataset(lr_dir, hr_dir)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        for lr_imgs, hr_imgs in dataloader:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            optimizer.zero_grad()
            outputs = model(lr_imgs)
            loss = criterion(outputs, hr_imgs)
            loss.backward()
            optimizer.step()
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")

    torch.save(model.state_dict(), 'models/srcnn_model.pth')

if __name__ == "__main__":
    lr_dir = '../data/processed_images'
    hr_dir = '../data/processed_images'
    train_model(lr_dir, hr_dir)
