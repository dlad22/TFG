from NCC import calculate_ncc
import scipy.io
import os

# Cargar las imágenes
images = scipy.io.loadmat('images.mat')

source = images['source']
target = images['target']
J0 = images['J0']


ncc = calculate_ncc(source, source)
print(f'NCC entre las imágenes: {ncc}')

# Calcular la NCC entre las imágenes
ncc = calculate_ncc(source, target)
print(f'NCC entre source y target: {ncc}')

# Calcular la NCC entre source y J0
ncc = calculate_ncc(source, J0)
print(f'NCC entre source y J0: {ncc}')

# Calcular la NCC entre target y J0
ncc = calculate_ncc(target, J0)
print(f'NCC entre target y J0: {ncc}')

