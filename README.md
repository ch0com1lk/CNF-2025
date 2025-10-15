# Reconstrucción Holográfica Precisa a partir de Patrones Marginales mediante Redes Neuronales

**Trabajo LXVIII-011184**  
Presentado para la Presentación de un Cartel en el Área de Óptica dentro del LXVIII Congreso Nacional de Física,  
Fecha: 16 de octubre de 2025,   
Lugar: Centro de Convenciones Estado de México,  
Organizado por la Sociedad Mexicana de Física.

## Reconstrucción Holográfica con Redes Neuronales

Este proyecto implementa un sistema de reconstrucción holográfica basado en redes neuronales que permite reconstruir hologramas completos a partir de patrones parciales de difracción.

## Requisitos

- Python 3.8 o superior  
- PyTorch  
- NumPy  
- Pillow  
- scikit-image

## Instalación

pip install torch torchvision numpy pillow scikit-image


## Estructura de Directorios

proyecto/
├── train_patterns/ # Patrones parciales de entrenamiento
│ ├── pattern_0_0.png # Primer patrón del primer conjunto
│ ├── pattern_0_1.png # Segundo patrón del primer conjunto
│ ├── pattern_0_2.png # Tercer patrón del primer conjunto
│ └── pattern_0_3.png # Cuarto patrón del primer conjunto
├── train_full/ # Hologramas completos de entrenamiento
│ └── hologram_0.png # Holograma completo correspondiente
├── val_patterns/ # Patrones parciales de validación
└── val_full/ # Hologramas completos de validación


## Preparación de Imágenes

1. Divide tus hologramas en 4 patrones parciales de 0.5×0.5 pulgadas.  
2. Nombra los patrones como `pattern_N_M.png` donde:  
   - N es el número de conjunto (0, 1, 2, ...)  
   - M es el número de patrón (0, 1, 2, 3)  
3. Coloca los patrones en las carpetas correspondientes.

## Uso

1. Prepara tus imágenes según la estructura anterior.  
2. Ejecuta el entrenamiento:

python hologram_reconstruction_3d.py


### El sistema

- Procesa patrones marginales de 0.5×0.5 pulgadas.  
- Usa una U-Net 3D con conexiones residuales.  
- Implementa métricas PSNR, SSIM y correlación cruzada.  
- Mantiene tiempos de procesamiento menores a 50 ms.  
- Logra precisión mayor al 99.2%.

## Uso con Imágenes de Prueba Propias

1. Divide tu holograma en 4 partes iguales.  
2. Guarda cada parte como `pattern_test_N.png` donde N = 0, 1, 2, 3.  
3. Coloca las imágenes en `val_patterns/`.  
4. El holograma completo correspondiente debe ir en `val_full/`.

El modelo guardará los mejores pesos en el archivo `best_model.pth`.

---

## Metodología

- Modulación espacial de fase mediante un modulador espacial de luz (SLM).  
- Red neuronal convolucional tridimensional U-Net con conexiones residuales para la reconstrucción.  
- Entrenamiento con función de pérdida que combina:  
  - Error cuadrático medio (MSE)  
  - Gradientes espaciales  
  - Coherencia de fase

## Validación Experimental

La reconstrucción se valida mediante interferometría digital y métricas cuantitativas como:  
- PSNR (Peak Signal-to-Noise Ratio)  
- SSIM (Structural Similarity Index)  
- Correlación cruzada normalizada

