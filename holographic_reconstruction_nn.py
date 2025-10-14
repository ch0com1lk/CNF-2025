import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision.transforms import ToTensor, Resize, Compose
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# 1. Simulación de la propagación de la luz y generación de hologramas
def create_hologram(image_tensor):
    """
    Simula la creación de un holograma a partir de una imagen de entrada.
    Utiliza la Transformada de Fourier 2D para simular la propagación de campo lejano (Fraunhofer).
    """
    # Asegurarse de que la imagen esté en el formato correcto (N, C, H, W)
    if image_tensor.ndim == 2:
        image_tensor = image_tensor.unsqueeze(0)  # Añadir dimensión de canal
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)  # Añadir dimensión de batch
        
    # La imagen se trata como la amplitud de un campo de luz en el plano del objeto
    object_field = image_tensor.to(torch.complex64)
    
    # Propagación al plano del holograma usando la Transformada de Fourier
    hologram_field = torch.fft.fftshift(torch.fft.fft2(object_field))
    
    # La intensidad es lo que se suele medir (se pierde la fase)
    hologram_intensity = torch.abs(hologram_field)**2
    
    # Normalizar para visualización y entrenamiento
    hologram_intensity = (hologram_intensity - hologram_intensity.min()) / (hologram_intensity.max() - hologram_intensity.min())
    
    return hologram_intensity

# 2. Arquitectura de la Red Neuronal (U-Net simplificada)
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
        output = torch.sigmoid(self.final_conv(d1)) # Sigmoid para normalizar la salida a [0, 1]
        return output

# 3. Preparación de datos con imágenes personalizadas
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(root_dir) 
                                   if os.path.isfile(os.path.join(root_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('L')  # Convertir a escala de grises
        if self.transform:
            image = self.transform(image)
        return image

def prepare_custom_data(image_dir, num_samples, img_size=64):
    transform = Compose([
        ToTensor(),
        Resize((img_size, img_size)),
    ])
    
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"No se encontró el directorio: '{image_dir}'")

    custom_dataset = CustomImageDataset(root_dir=image_dir, transform=transform)
    
    if len(custom_dataset) == 0:
        raise FileNotFoundError(f"No se encontraron imágenes en el directorio: '{image_dir}'")

    original_images = []
    holograms = []
    
    num_to_process = min(num_samples, len(custom_dataset))
    print(f"Procesando {num_to_process} imágenes de '{image_dir}'...")

    for i in range(num_to_process):
        image = custom_dataset[i]
        # Asegurarse de que la imagen tenga la forma correcta (C, H, W)
        if image.dim() == 2:
            image = image.unsqueeze(0)
        if image.mean() > 0.01:  # Evitar imágenes casi vacías
            hologram = create_hologram(image)
            # Asegurarse de que el holograma tenga la forma correcta (C, H, W)
            if hologram.dim() == 4:
                hologram = hologram.squeeze(0)
            original_images.append(image)
            holograms.append(hologram)
    
    if not original_images:
        raise ValueError("No se pudieron procesar las imágenes. Asegúrate de que no estén completamente negras o vacías.")

    original_images = torch.stack(original_images)
    holograms = torch.stack(holograms)
    
    return TensorDataset(holograms, original_images)


if __name__ == '__main__':
    # Parámetros
    IMG_SIZE = 64
    NUM_SAMPLES = 2000
    BATCH_SIZE = 4
    EPOCHS = 50
    LEARNING_RATE = 5e-4
    IMAGE_DIR = '/Users/macbookpro/Downloads/imagenesdeprueba'  # Directorio actualizado
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {DEVICE}")

    try:
        print("Preparando datos desde el directorio personalizado...")
        dataset = prepare_custom_data(image_dir=IMAGE_DIR, num_samples=NUM_SAMPLES, img_size=IMG_SIZE)
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        exit()
    
    model = HolographicUNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    print("Iniciando entrenamiento...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_idx, (hologram_batch, image_batch) in enumerate(train_loader):
            hologram_batch, image_batch = hologram_batch.to(DEVICE), image_batch.to(DEVICE)
            
            reconstructed_batch = model(hologram_batch)
            loss = criterion(reconstructed_batch, image_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}")

    print("Entrenamiento finalizado. Guardando modelo...")
    # Guardar el modelo entrenado
    torch.save(model.state_dict(), 'holographic_model.pth')
    print("Modelo guardado como 'holographic_model.pth'")
    
    print("Mostrando resultados...")
    model.eval()
    with torch.no_grad():
        holograms_sample, images_sample = next(iter(train_loader))
        holograms_sample = holograms_sample.to(DEVICE)
        
        reconstructed_images = model(holograms_sample).cpu()

        n_examples = min(5, len(images_sample))
        if n_examples > 0:
            plt.figure(figsize=(15, 5 * n_examples))
            for i in range(n_examples):
                # Imagen Original
                plt.subplot(n_examples, 3, i * 3 + 1)
                plt.imshow(images_sample[i].squeeze(), cmap='gray')
                plt.title("Original")
                plt.axis('off')

                # Holograma
                plt.subplot(n_examples, 3, i * 3 + 2)
                hologram_display = torch.log(holograms_sample[i].cpu().squeeze() + 1e-9)
                plt.imshow(hologram_display, cmap='viridis')
                plt.title("Holograma (Entrada)")
                plt.axis('off')

                # Imagen Reconstruida
                plt.subplot(n_examples, 3, i * 3 + 3)
                plt.imshow(reconstructed_images[i].squeeze(), cmap='gray')
                plt.title("Reconstruida por la Red")
                plt.axis('off')
            
            plt.tight_layout()
            plt.show()
        else:
            print("No hay imágenes para mostrar.")