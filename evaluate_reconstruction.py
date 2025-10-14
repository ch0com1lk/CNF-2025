import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
from hologram_reconstruction_2d import HologramUNet, HologramDataset
from torchvision import transforms
import time

def calculate_metrics(original, reconstructed):
    """Calcula PSNR, SSIM y correlación cruzada normalizada."""
    original = original.squeeze().cpu().numpy()
    reconstructed = reconstructed.squeeze().cpu().numpy()
    
    # PSNR
    psnr_value = psnr(original, reconstructed, data_range=1.0)
    
    # SSIM
    ssim_value = ssim(original, reconstructed, data_range=1.0)
    
    # Correlación cruzada normalizada
    correlation = np.corrcoef(original.flatten(), reconstructed.flatten())[0,1]
    
    return psnr_value, ssim_value, correlation

def evaluate_model(model, dataloader, device):
    """Evalúa el modelo y genera visualizaciones detalladas."""
    model.eval()
    
    # Métricas acumuladas
    metrics = {'psnr': [], 'ssim': [], 'correlation': [], 'time': []}
    
    with torch.no_grad():
        for patterns, target in dataloader:
            patterns, target = patterns.to(device), target.to(device)
            
            # Medir tiempo de procesamiento
            start_time = time.time()
            output = model(patterns)
            process_time = (time.time() - start_time) * 1000  # convertir a ms
            
            # Calcular métricas
            psnr_val, ssim_val, corr = calculate_metrics(target, output)
            
            metrics['psnr'].append(psnr_val)
            metrics['ssim'].append(ssim_val)
            metrics['correlation'].append(corr)
            metrics['time'].append(process_time)
            
            # Visualización detallada
            plt.figure(figsize=(20, 10))
            
            # Patrones originales
            for i in range(4):
                plt.subplot(2, 4, i+1)
                plt.imshow(patterns[0][i].cpu(), cmap='gray')
                plt.title(f'Patrón {i+1}')
                plt.axis('off')
            
            # Holograma original
            plt.subplot(2, 4, 5)
            plt.imshow(target[0].cpu().squeeze(), cmap='gray')
            plt.title('Holograma Original')
            plt.axis('off')
            
            # Reconstrucción
            plt.subplot(2, 4, 6)
            plt.imshow(output[0].cpu().squeeze(), cmap='gray')
            plt.title('Reconstrucción')
            plt.axis('off')
            
            # Diferencia
            plt.subplot(2, 4, 7)
            diff = torch.abs(target - output)
            plt.imshow(diff[0].cpu().squeeze(), cmap='hot')
            plt.title('Mapa de Diferencias')
            plt.colorbar()
            plt.axis('off')
            
            # Perfil de intensidad
            plt.subplot(2, 4, 8)
            center_line_orig = target[0, 0, target.shape[2]//2, :].cpu()
            center_line_recon = output[0, 0, output.shape[2]//2, :].cpu()
            plt.plot(center_line_orig, label='Original')
            plt.plot(center_line_recon, label='Reconstruido')
            plt.title('Perfil de Intensidad')
            plt.legend()
            
            plt.suptitle(f'Análisis Detallado\nPSNR: {psnr_val:.2f}dB, SSIM: {ssim_val:.4f}\n'
                        f'Correlación: {corr:.4f}, Tiempo: {process_time:.1f}ms')
            
            plt.tight_layout()
            plt.savefig('analisis_detallado.png', dpi=300, bbox_inches='tight')
            plt.show()
            
    # Resumen de métricas
    print("\nResumen de Métricas:")
    print(f"PSNR promedio: {np.mean(metrics['psnr']):.2f} dB")
    print(f"SSIM promedio: {np.mean(metrics['ssim']):.4f}")
    print(f"Correlación promedio: {np.mean(metrics['correlation']):.4f}")
    print(f"Tiempo de procesamiento promedio: {np.mean(metrics['time']):.1f} ms")
    
    return metrics

if __name__ == '__main__':
    # Configuración
    device = torch.device('cpu')
    
    # Cargar el modelo entrenado
    model = HologramUNet().to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Preparar datos
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    dataset = HologramDataset(
        pattern_dir='/Users/macbookpro/Downloads/train_patterns',
        full_dir='/Users/macbookpro/Downloads/train_full',
        transform=transform
    )
    
    dataloader = DataLoader(dataset, batch_size=1)
    
    print("Iniciando evaluación detallada...")
    metrics = evaluate_model(model, dataloader, device)
    
    # Guardar métricas en un archivo
    with open('metricas_reconstruccion.txt', 'w') as f:
        f.write("Métricas de Reconstrucción Holográfica\n")
        f.write("=====================================\n\n")
        f.write(f"PSNR promedio: {np.mean(metrics['psnr']):.2f} dB\n")
        f.write(f"SSIM promedio: {np.mean(metrics['ssim']):.4f}\n")
        f.write(f"Correlación promedio: {np.mean(metrics['correlation']):.4f}\n")
        f.write(f"Tiempo de procesamiento promedio: {np.mean(metrics['time']):.1f} ms\n\n")
        f.write("Nota: El análisis detallado se ha guardado en 'analisis_detallado.png'")