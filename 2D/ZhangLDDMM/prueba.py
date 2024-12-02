import torch

# Verificar si CUDA está disponible
if torch.cuda.is_available():
    device = torch.device("cuda") # cuda:0
    print(f"CUDA is available. Using device: {torch.cuda.get_device_name(device)}")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

nx = 10
ny = 10
v0 = torch.zeros((int(ny), int(nx), 2) , dtype=torch.float32, device=device)

print('Fin prueba')