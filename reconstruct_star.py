import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor, Resize, Compose
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

class HologramReconstructor(nn.Module):
    def __init__(self):
        super(HologramReconstructor, self).__init__()
        self.net = nn.Sequential(
            # Encoder
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            
            # Decoder
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def create_hologram(image):
    """Simula la creación de un holograma."""
    # Asegurar que la imagen esté en el formato correcto
    if len(image.shape) == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif len(image.shape) == 3:
        image = image.unsqueeze(0)

    # Convertir a campo complejo
    field = image.to(torch.complex64)
    
    # Transformada de Fourier
    hologram = torch.fft.fft2(field)
    hologram = torch.fft.fftshift(hologram)
    
    # Calcular intensidad
    intensity = torch.abs(hologram)**2
    
    # Normalizar
    intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min())
    
    return intensity

def load_image(path, size=128):
    """Carga y preprocesa una imagen."""
    transform = Compose([
        ToTensor(),
        Resize((size, size))
    ])
    
    # Cargar imagen y convertir a escala de grises
    image = Image.open(path).convert('L')
    image = transform(image)
    
    # Normalizar
    image = (image - image.min()) / (image.max() - image.min())
    
    return image.unsqueeze(0)  # Añadir dimensión de batch

if __name__ == '__main__':
    # Configuración
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMAGE_PATH = '/Users/macbookpro/Downloads/estrellaprueba/estrellasprueba18.jpeg'
    IMG_SIZE = 128
    NUM_EPOCHS = 1000
    LEARNING_RATE = 1e-3
    
    print(f"Usando dispositivo: {DEVICE}")
    
    # Cargar imagen
    print("Cargando imagen...")
    original = load_image(IMAGE_PATH, IMG_SIZE).to(DEVICE)
    
    # Crear holograma
    print("Generando holograma...")
    hologram = create_hologram(original)
    
    # Crear y entrenar el modelo
    model = HologramReconstructor().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    print("Iniciando entrenamiento...")
    losses = []
    
    for epoch in range(NUM_EPOCHS):
        # Forward pass
        reconstructed = model(hologram)
        reconstructed_hologram = create_hologram(reconstructed)
        
        # Calcular pérdida
        loss = criterion(reconstructed_hologram, hologram)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 100 == 0:
            print(f"Época [{epoch+1}/{NUM_EPOCHS}], Pérdida: {loss.item():.6f}")
    
    print("Entrenamiento completado. Generando visualización...")
    
    # Visualización
    with torch.no_grad():
        model.eval()
        final_reconstruction = model(hologram)
        
        plt.figure(figsize=(15, 5))
        
        # Imagen original
        plt.subplot(131)
        plt.imshow(original.cpu().squeeze(), cmap='gray')
        plt.title('Imagen Original')
        plt.axis('off')
        
        # Holograma
        plt.subplot(132)
        plt.imshow(hologram.cpu().squeeze().real, cmap='viridis')
        plt.title('Holograma')
        plt.axis('off')
        
        # Reconstrucción
        plt.subplot(133)
        plt.imshow(final_reconstruction.cpu().squeeze(), cmap='gray')
        plt.title('Reconstrucción')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Guardar la reconstrucción
        reconstruction_image = Image.fromarray((final_reconstruction.cpu().squeeze().numpy() * 255).astype(np.uint8))
        reconstruction_image.save('estrella_reconstruida.png')
        print("Reconstrucción guardada como 'estrella_reconstruida.png'")