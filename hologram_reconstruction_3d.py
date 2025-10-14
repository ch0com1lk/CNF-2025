import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import time

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

class Encoder3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder3DBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.res = ResidualBlock(out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        skip = self.res(x)
        x = self.pool(skip)
        return x, skip

class Decoder3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder3DBlock, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.res = ResidualBlock(out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.res(x)
        return x

class HologramUNet3D(nn.Module):
    def __init__(self):
        super(HologramUNet3D, self).__init__()
        self.enc1 = Encoder3DBlock(1, 64)
        self.enc2 = Encoder3DBlock(64, 128)
        self.enc3 = Encoder3DBlock(128, 256)
        
        self.bridge = ResidualBlock(256)
        
        self.dec3 = Decoder3DBlock(256, 128)
        self.dec2 = Decoder3DBlock(128, 64)
        self.dec1 = Decoder3DBlock(64, 32)
        
        self.final = nn.Conv3d(32, 1, kernel_size=1)

    def forward(self, x):
        x1, skip1 = self.enc1(x)
        x2, skip2 = self.enc2(x1)
        x3, skip3 = self.enc3(x2)
        
        x = self.bridge(x3)
        
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        
        return self.final(x)

class HologramDataset(Dataset):
    def __init__(self, pattern_dir, full_dir, transform=None):
        self.pattern_dir = pattern_dir
        self.full_dir = full_dir
        self.transform = transform
        self.pattern_files = sorted([f for f in os.listdir(pattern_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.full_files = sorted([f for f in os.listdir(full_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.full_files)

    def __getitem__(self, idx):
        patterns = []
        for i in range(4):
            pattern_path = os.path.join(self.pattern_dir, f'pattern_{idx}_{i}.png')
            pattern = Image.open(pattern_path).convert('L')
            if self.transform:
                pattern = self.transform(pattern)
            patterns.append(pattern)
        
        # Reorganizar los patrones en un tensor 3D [1, 2, 2, H, W]
        patterns = torch.stack(patterns)
        patterns = patterns.view(1, 2, 2, patterns.shape[1], patterns.shape[2])
        
        full_path = os.path.join(self.full_dir, self.full_files[idx])
        full_hologram = Image.open(full_path).convert('L')
        if self.transform:
            full_hologram = self.transform(full_hologram)
        
        return patterns, full_hologram.unsqueeze(0)

def compute_metrics(pred, target):
    pred_np = pred.cpu().numpy().squeeze()
    target_np = target.cpu().numpy().squeeze()
    
    psnr_val = psnr(target_np, pred_np)
    ssim_val = ssim(target_np, pred_np)
    
    cross_corr = np.corrcoef(pred_np.flatten(), target_np.flatten())[0,1]
    
    return psnr_val, ssim_val, cross_corr

def custom_loss(pred, target, alpha=1.0, beta=0.5, gamma=0.3):
    mse_loss = F.mse_loss(pred, target)
    
    dx = pred[:,:,1:,:] - pred[:,:,:-1,:]
    dy = pred[:,:,:,1:] - pred[:,:,:,:-1]
    gradient_loss = torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))
    
    fft_pred = torch.fft.fft2(pred)
    fft_target = torch.fft.fft2(target)
    phase_loss = F.mse_loss(torch.angle(fft_pred), torch.angle(fft_target))
    
    return mse_loss + alpha * gradient_loss + beta * phase_loss

def train_model(model, train_loader, val_loader, epochs=100, device='cpu'):  # Más épocas para mejor convergencia
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        for patterns, target in train_loader:
            patterns, target = patterns.to(device), target.to(device)
            
            start_time = time.time()
            
            optimizer.zero_grad()
            output = model(patterns)
            loss = custom_loss(output, target)
            loss.backward()
            optimizer.step()
            
            process_time = time.time() - start_time
            
            if process_time > 0.05:  # 50ms threshold
                print(f'Warning: Processing time ({process_time*1000:.1f}ms) exceeds 50ms target')
        
        model.eval()
        val_loss = 0
        metrics = {'psnr': 0, 'ssim': 0, 'cross_corr': 0}
        with torch.no_grad():
            for patterns, target in val_loader:
                patterns, target = patterns.to(device), target.to(device)
                output = model(patterns)
                val_loss += custom_loss(output, target).item()
                
                psnr_val, ssim_val, cross_corr = compute_metrics(output, target)
                metrics['psnr'] += psnr_val
                metrics['ssim'] += ssim_val
                metrics['cross_corr'] += cross_corr
        
        val_loss /= len(val_loader)
        for key in metrics:
            metrics[key] /= len(val_loader)
        
        print(f'Epoch {epoch+1}, Val Loss: {val_loss:.4f}, PSNR: {metrics["psnr"]:.2f}, '
              f'SSIM: {metrics["ssim"]:.4f}, CrossCorr: {metrics["cross_corr"]:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

if __name__ == '__main__':
    # Configuración
    device = torch.device('cpu')
    print(f"Usando dispositivo: {device}")
    
    # Transformaciones para las imágenes
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Tamaño más manejable
        transforms.ToTensor(),
    ])
    
    # Crear dataset solo con la imagen de la estrella
    train_dataset = HologramDataset(
        pattern_dir='/Users/macbookpro/Downloads/train_patterns',
        full_dir='/Users/macbookpro/Downloads/train_full',
        transform=transform
    )
    
    # Usar el mismo dataset para validación
    val_dataset = train_dataset
    
    # Dataloaders - batch_size=1 ya que solo tenemos una imagen
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)
    
    # Modelo
    model = HologramUNet3D().to(device)
    
    # Entrenamiento
    train_model(model, train_loader, val_loader)