import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Resize, Compose
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Arquitectura de la Red Neuronal (U-Net simplificada)
class HolographicUNet(nn.Module):
    def __init__(self):
        super(HolographicUNet, self).__init__()
        # Codificador
        self.enc1 = self.conv_block(1, 64)
        self.enc2 = self.conv_block(64, 128)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = self.conv_block(128, 256)
        
        # Decodificador
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        # Capa final
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Codificador
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        
        # Bottleneck
        b = self.bottleneck(p2)
        
        # Decodificador
        d2 = self.upconv2(b)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)
        
        # Salida
        output = torch.sigmoid(self.final_conv(d1))
        return output

def create_hologram(image_tensor):
    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0)
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
        
    object_field = image_tensor.to(torch.complex64)
    hologram_field = torch.fft.fftshift(torch.fft.fft2(object_field))
    hologram_intensity = torch.abs(hologram_field)**2
    hologram_intensity = (hologram_intensity - hologram_intensity.min()) / (hologram_intensity.max() - hologram_intensity.min())
    
    return hologram_intensity

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(root_dir) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image, self.image_files[idx]

def evaluate_images(model, image_dir, img_size=64, batch_size=1):
    transform = Compose([
        ToTensor(),
        Resize((img_size, img_size))
    ])
    
    dataset = ImageDataset(root_dir=image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    device = next(model.parameters()).device
    model.eval()
    
    with torch.no_grad():
        for images, filenames in dataloader:
            # Procesar cada imagen
            for image, filename in zip(images, filenames):
                # Preparar la imagen
                if image.dim() == 2:
                    image = image.unsqueeze(0)
                
                # Crear el holograma
                hologram = create_hologram(image)
                if hologram.dim() == 4:
                    hologram = hologram.squeeze(0)
                
                # Reconstruir la imagen
                hologram_input = hologram.to(device)
                reconstructed = model(hologram_input)
                
                # Mostrar resultados
                plt.figure(figsize=(15, 5))
                
                # Imagen Original
                plt.subplot(1, 3, 1)
                plt.imshow(image.squeeze().cpu(), cmap='gray')
                plt.title(f"Original: {filename}")
                plt.axis('off')
                
                # Holograma
                plt.subplot(1, 3, 2)
                hologram_display = torch.log(hologram.squeeze().cpu() + 1e-9)
                plt.imshow(hologram_display, cmap='viridis')
                plt.title("Holograma")
                plt.axis('off')
                
                # Imagen Reconstruida
                plt.subplot(1, 3, 3)
                plt.imshow(reconstructed.squeeze().cpu(), cmap='gray')
                plt.title("Reconstruida")
                plt.axis('off')
                
                plt.tight_layout()
                plt.show()

if __name__ == '__main__':
    # Parámetros
    IMG_SIZE = 64
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TEST_DIR = '/Users/macbookpro/Downloads/estrellaprueba'
    
    # Cargar el modelo entrenado
    model = HolographicUNet().to(DEVICE)
    # Intentar cargar los pesos del modelo desde el archivo guardado
    try:
        model.load_state_dict(torch.load('holographic_model.pth'))
        print("Modelo cargado exitosamente.")
    except:
        print("No se encontró un modelo guardado. Por favor, entrena el modelo primero.")
        exit()
    
    # Evaluar las imágenes
    print(f"Evaluando imágenes en {TEST_DIR}...")
    evaluate_images(model, TEST_DIR, IMG_SIZE)