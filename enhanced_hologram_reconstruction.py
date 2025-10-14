import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = torch.relu(x)
        return x

class EnhancedUNet(nn.Module):
    def __init__(self, in_channels=4):
        super(EnhancedUNet, self).__init__()
        
        # Encoder
        self.enc1 = self._make_layer(in_channels, 64)
        self.enc2 = self._make_layer(64, 128)
        self.enc3 = self._make_layer(128, 256)
        self.enc4 = self._make_layer(256, 512)
        
        # Bridge
        self.bridge = nn.Sequential(
            ResidualBlock(512),
            ResidualBlock(512)
        )
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = self._make_layer(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self._make_layer(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self._make_layer(128, 64)
        
        # Final
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(0.1)
        self.pool = nn.MaxPool2d(2, 2)

    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(out_channels)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bridge
        b = self.bridge(self.pool(e4))
        b = self.dropout(b)
        
        # Decoder
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        return self.final(d2)

class HologramDataset(Dataset):
    def __init__(self, pattern_dir, full_dir, transform=None):
        self.pattern_dir = pattern_dir
        self.full_dir = full_dir
        self.transform = transform
        self.full_files = sorted([f for f in os.listdir(full_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.full_files)

    def __getitem__(self, idx):
        patterns = []
        for i in range(4):
            pattern_path = os.path.join(self.pattern_dir, f'pattern_0_{i}.png')
            pattern = Image.open(pattern_path).convert('L')
            if self.transform:
                pattern = self.transform(pattern)
            patterns.append(pattern)
        
        patterns = torch.cat(patterns, dim=0)
        
        full_path = os.path.join(self.full_dir, self.full_files[idx])
        full_hologram = Image.open(full_path).convert('L')
        if self.transform:
            full_hologram = self.transform(full_hologram)
        
        return patterns, full_hologram

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return torch.mean(focal_loss)

def compute_metrics(pred, target):
    pred_np = pred.cpu().numpy().squeeze()
    target_np = target.cpu().numpy().squeeze()
    
    # Normalizar a [0, 1]
    pred_np = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min())
    target_np = (target_np - target_np.min()) / (target_np.max() - target_np.min())
    
    # Calcular métricas
    psnr_val = psnr(target_np, pred_np)
    ssim_val = ssim(target_np, pred_np)
    mse = np.mean((target_np - pred_np) ** 2)
    cross_corr = np.corrcoef(pred_np.flatten(), target_np.flatten())[0,1]
    
    return {
        'PSNR': psnr_val,
        'SSIM': ssim_val,
        'MSE': mse,
        'CrossCorr': cross_corr
    }

def train_model(model, train_loader, val_loader, epochs=500, device='cpu'):
    criterion = FocalLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
    
    best_loss = float('inf')
    no_improve = 0
    history = {'train_loss': [], 'val_metrics': []}
    
    for epoch in range(epochs):
        # Entrenamiento
        model.train()
        train_loss = 0
        for patterns, target in train_loader:
            patterns, target = patterns.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(patterns)
            loss = criterion(output, target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validación
        model.eval()
        val_loss = 0
        all_metrics = []
        with torch.no_grad():
            for patterns, target in val_loader:
                patterns, target = patterns.to(device), target.to(device)
                output = model(patterns)
                val_loss += criterion(output, target).item()
                
                metrics = compute_metrics(output, target)
                all_metrics.append(metrics)
        
        avg_val_loss = val_loss / len(val_loader)
        avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
        history['val_metrics'].append(avg_metrics)
        
        # Actualizar learning rate
        scheduler.step(avg_val_loss)
        
        # Guardar mejor modelo
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'best_model_enhanced.pth')
            no_improve = 0
        else:
            no_improve += 1
        
        if epoch % 10 == 0:
            print(f'Época [{epoch+1}/{epochs}]')
            print(f'Pérdida Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f}')
            print(f'Métricas: PSNR={avg_metrics["PSNR"]:.2f}, SSIM={avg_metrics["SSIM"]:.4f}, CrossCorr={avg_metrics["CrossCorr"]:.4f}')
            
            # Visualización
            model.eval()
            with torch.no_grad():
                for patterns, target in val_loader:
                    patterns, target = patterns.to(device), target.to(device)
                    output = model(patterns)
                    
                    plt.figure(figsize=(15, 10))
                    
                    # Patrones originales
                    for i in range(4):
                        plt.subplot(3, 4, i+1)
                        plt.imshow(patterns[0][i].cpu(), cmap='gray')
                        plt.title(f'Patrón {i+1}')
                        plt.axis('off')
                    
                    # Holograma original y reconstruido
                    plt.subplot(3, 2, 4)
                    plt.imshow(target[0].cpu().squeeze(), cmap='gray')
                    plt.title('Original')
                    plt.axis('off')
                    
                    plt.subplot(3, 2, 5)
                    plt.imshow(output[0].cpu().squeeze(), cmap='gray')
                    plt.title(f'Reconstrucción\nPSNR={avg_metrics["PSNR"]:.2f}')
                    plt.axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(f'reconstruction_epoch_{epoch+1}.png')
                    plt.close()
                    break
        
        # Early stopping
        if no_improve >= 50:
            print("Early stopping triggered")
            break
    
    return history

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
    
    model = EnhancedUNet().to(device)
    
    print("Iniciando entrenamiento mejorado...")
    history = train_model(model, train_loader, val_loader)
    
    # Guardar historial de entrenamiento
    np.save('training_history.npy', history)
    
    # Gráfica final de métricas
    epochs = len(history['train_loss'])
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.plot(history['train_loss'])
    plt.title('Pérdida de Entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    
    plt.subplot(132)
    psnr_values = [m['PSNR'] for m in history['val_metrics']]
    plt.plot(psnr_values)
    plt.title('PSNR')
    plt.xlabel('Época')
    
    plt.subplot(133)
    ssim_values = [m['SSIM'] for m in history['val_metrics']]
    plt.plot(ssim_values)
    plt.title('SSIM')
    plt.xlabel('Época')
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()