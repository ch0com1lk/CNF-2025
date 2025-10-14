from PIL import Image
import os
import numpy as np
import torch
from torchvision import transforms

def prepare_patterns(image_path, output_dir, pattern_size=(256, 256)):
    # Cargar la imagen
    img = Image.open(image_path).convert('L')
    
    # Redimensionar a 512x512 para tener una resolución adecuada
    img = img.resize((512, 512))
    
    # Convertir a array de numpy
    img_array = np.array(img)
    
    # Dividir en 4 patrones (2x2)
    h, w = img_array.shape
    h2, w2 = h//2, w//2
    
    patterns = [
        img_array[:h2, :w2],     # Superior izquierdo
        img_array[:h2, w2:],     # Superior derecho
        img_array[h2:, :w2],     # Inferior izquierdo
        img_array[h2:, w2:]      # Inferior derecho
    ]
    
    # Crear directorios si no existen
    os.makedirs(output_dir + '/train_patterns', exist_ok=True)
    os.makedirs(output_dir + '/train_full', exist_ok=True)
    
    # Guardar los patrones
    for i, pattern in enumerate(patterns):
        pattern_img = Image.fromarray(pattern)
        pattern_img = pattern_img.resize(pattern_size)
        pattern_img.save(f'{output_dir}/train_patterns/pattern_0_{i}.png')
    
    # Guardar la imagen completa
    img = img.resize((512, 512))
    img.save(f'{output_dir}/train_full/hologram_0.png')
    
    print(f"Patrones guardados en {output_dir}/train_patterns/")
    print(f"Holograma completo guardado en {output_dir}/train_full/")
    return patterns

if __name__ == '__main__':
    # Rutas
    input_image = '/Users/macbookpro/Downloads/estrellaprueba/estrellasprueba18.jpeg'
    output_dir = '/Users/macbookpro/Downloads'
    
    # Preparar los patrones
    patterns = prepare_patterns(input_image, output_dir)
    
    print("\nProceso completado. Ahora puedes ejecutar hologram_reconstruction_3d.py")