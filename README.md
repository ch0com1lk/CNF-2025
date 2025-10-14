# CNF-2025
# Reconstrucción Holográfica Precisa a partir de Patrones Marginales mediante Redes Neuronales

**Trabajo LXVIII-011184**  
Presentado en Póster en el Área de Óptica del LXVIII Congreso Nacional de Física  
Fecha: 16 de octubre de 2025  
Lugar: Centro de Convenciones Estado de México  
Organizado por Sociedad Mexicana de Física.

## Descripción del Trabajo

Este proyecto aborda el desafío de reconstruir hologramas completos con alta resolución a partir de patrones parciales de difracción, fundamental para el desarrollo de sistemas ópticos compactos y de precisión submicrométrica.

Se propone un enfoque híbrido óptico-computacional que genera un holograma completo de 1×1 pulgada, combinando cuatro patrones marginales de 0.5 pulgadas cuadradas cada uno.

## Metodología

- Modulación espacial de fase mediante un modulador espacial de luz (SLM).
- Red neuronal convolucional tridimensional U-Net con conexiones residuales para la reconstrucción.
- Entrenamiento con función de pérdida que combina:
  - Error cuadrático medio (MSE)
  - Gradientes espaciales
  - Coherencia de fase

## Validación Experimental

La reconstrucción se valida mediante interferometría digital y métricas cuantitativas, como:  
- PSNR (Peak Signal-to-Noise Ratio)  
- SSIM (Structural Similarity Index)  
- Correlación cruzada normalizada
