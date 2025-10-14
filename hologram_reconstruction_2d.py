import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class HologramUNet(nn.Module):
    def __init__(self):
        super(HologramUNet, self).__init__()
        
        # Encoder
        self.enc1 = ConvBlock(4, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        
        # Final
        self.final = nn.Conv2d(64, 1, 1)
        
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Decoder
        d3 = self.up3(e3)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        
        out = torch.sigmoid(self.final(d2))
        return out

class HologramDataset(Dataset):
    def __init__(self, pattern_dir, full_dir, transform=None):
        self.pattern_dir = pattern_dir
        self.full_dir = full_dir
        self.transform = transform
        self.full_files = sorted([f for f in os.listdir(full_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.full_files)

    def __getitem__(self, idx):
        # Cargar los 4 patrones
        patterns = []
        for i in range(4):
            pattern_path = os.path.join(self.pattern_dir, f'pattern_0_{i}.png')
            pattern = Image.open(pattern_path).convert('L')
            if self.transform:
                pattern = self.transform(pattern)
            patterns.append(pattern)
        
        patterns = torch.cat(patterns, dim=0)  # Concatenar en el canal
        
        # Cargar el holograma completo
        full_path = os.path.join(self.full_dir, self.full_files[idx])
        full_hologram = Image.open(full_path).convert('L')
        if self.transform:
            full_hologram = self.transform(full_hologram)
        
        return patterns, full_hologram

def train_model(model, train_loader, val_loader, epochs=100, device='cpu'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for patterns, target in train_loader:
            patterns, target = patterns.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(patterns)
            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_loss = train_loss / len(train_loader)
        print(f'Época [{epoch+1}/{epochs}], Pérdida: {avg_loss:.6f}')
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_model.pth')
            
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                for patterns, target in val_loader:
                    patterns, target = patterns.to(device), target.to(device)
                    output = model(patterns)
                    
                    # Visualizar resultados
                    plt.figure(figsize=(15, 5))
                    
                    plt.subplot(131)
                    plt.imshow(patterns[0][0].cpu(), cmap='gray')
                    plt.title('Patrón 1 de 4')
                    plt.axis('off')
                    
                    plt.subplot(132)
                    plt.imshow(target[0].cpu().squeeze(), cmap='gray')
                    plt.title('Original')
                    plt.axis('off')
                    
                    plt.subplot(133)
                    plt.imshow(output[0].cpu().squeeze(), cmap='gray')
                    plt.title(f'Reconstrucción (Época {epoch+1})')
                    plt.axis('off')
                    
                    plt.tight_layout()
                    plt.show()
                    break

if __name__ == '__main__':
    device = torch.device('cpu')
    print(f"Usando dispositivo: {device}")
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    dataset = HologramDataset(
        pattern_dir='/Users/macbookpro/Downloads/train_patterns',
        full_dir='/Users/macbookpro/Downloads/train_full',
        transform=transform
    )
    
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=1)
    
    model = HologramUNet().to(device)
    
    print("Iniciando entrenamiento...")
    train_model(model, train_loader, val_loader)