import torch

def calculate_ncc(image1, image2):
    """
    Calcula la Normalized Cross-Correlation (NCC) entre dos imágenes 3D usando PyTorch.
    
    Parámetros:
    image1, image2: tensores 3D de PyTorch con las dos imágenes a comparar.
    
    Retorna:
    ncc: Valor de NCC entre -1 y 1.
    """
    # Asegurarse de que las imágenes tengan la misma forma
    assert image1.shape == image2.shape, "Las imágenes deben tener la misma forma"

    # Pasar imagenes a torch si no lo están
    if not torch.is_tensor(image1):
        image1 = torch.tensor(image1)
    if not torch.is_tensor(image2):
        image2 = torch.tensor(image2)
    
    # Convertir las imágenes a flotantes para evitar problemas de desbordamiento
    image1 = image1.to(torch.float32)
    image2 = image2.to(torch.float32)
    
    # Calcular las medias de las imágenes
    mean1 = torch.mean(image1)
    mean2 = torch.mean(image2)
    
    # Restar las medias a las imágenes
    image1_mean_sub = image1 - mean1
    image2_mean_sub = image2 - mean2
    
    # Calcular el numerador de la NCC
    numerator = torch.sum(image1_mean_sub * image2_mean_sub)
    
    # Calcular los denominadores de la NCC
    denominator1 = torch.sqrt(torch.sum(image1_mean_sub ** 2))
    denominator2 = torch.sqrt(torch.sum(image2_mean_sub ** 2))
    
    # Calcular la NCC
    ncc = numerator / (denominator1 * denominator2)
    
    return ncc

