# @author: Zongyi Li
# This file is the Fourier Neural Operator for 3D problem such as the Navier-Stokes equation discussed in Section 5.3 in the [paper](https://arxiv.org/pdf/2010.08895.pdf),
# which takes the 3D spatial + 1D temporal equation directly as a 4D problem


# GitHub issues for reproducibility 19 Feb 2024
# https://github.com/neuraloperator/neuraloperator/issues/57

import numpy as np

import os
import scipy.io

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from utilities3 import *

import operator
from functools import reduce
from functools import partial

from timeit import default_timer

from Adam import Adam

import time
import subprocess

torch.manual_seed(0)
np.random.seed(0)

bigGPU = True

def get_gpu_info():
    # Ejecutar el comando nvidia-smi
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,temperature.gpu', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
    # Convertir la salida en una lista de strings
    output = result.stdout.decode('utf-8').strip().split('\n')
    
    for line in output:
        fields = line.split(',')
        print(f"GPU ID: {fields[0]}")
        print(f"Nombre: {fields[1]}")
        print(f"Uso de la GPU: {fields[2]}%")
        print(f"Uso de la memoria: {fields[3]}%")
        print(f"Memoria total: {fields[4]} MB")
        print(f"Memoria libre: {fields[5]} MB")
        print(f"Memoria utilizada: {fields[6]} MB")
        print(f"Temperatura: {fields[7]} C")
        print("-" * 20)

        # break

################################################################
# 4d fourier layers
################################################################

class Conv4d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv4d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        batch_size, channels, dim1, dim2, dim3, dim4 = x.shape
        
        # Reshape input to apply 2D convolutions
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(batch_size * dim1 * dim2, channels, dim3, dim4)
        
        x = self.conv2d(x)
        
        # Reshape back to 4D
        out_dim3, out_dim4 = x.shape[-2], x.shape[-1]
        x = x.view(batch_size, dim1, dim2, self.out_channels, out_dim3, out_dim4)
        x = x.permute(0, 3, 1, 2, 4, 5).contiguous()
        
        return x
    
class BatchNorm4d(nn.Module):
    def __init__(self, num_features):
        super(BatchNorm4d, self).__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm3d(num_features)

    def forward(self, x):
        batch_size, channels, dim1, dim2, dim3, dim4 = x.shape
        
        # Reformatear la entrada para aplicar BatchNorm3d
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(batch_size * dim1 * dim2, channels, dim3, dim4)
        
        x = self.bn(x)
        
        # Reformatear de nuevo a 4D
        x = x.view(batch_size, dim1, dim2, channels, dim3, dim4)
        x = x.permute(0, 3, 1, 2, 4, 5).contiguous()
        
        return x

class SpectralConv4d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, modes4):
        super(SpectralConv4d, self).__init__()

        """
        4D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3
        self.modes4 = modes4

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul4d(self, input, weights):
        # (batch, in_channel, x,y,z,t ), (in_channel, out_channel, x,y,z,t) -> (batch, out_channel, x,y,z,t)
        return torch.einsum("bixyzt,ioxyzt->boxyzt", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-4,-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-4), x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3, :self.modes4] = \
            self.compl_mul4d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3, :self.modes4], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3, :self.modes4] = \
            self.compl_mul4d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3, :self.modes4], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3, :self.modes4] = \
            self.compl_mul4d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3, :self.modes4], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3, :self.modes4] = \
            self.compl_mul4d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3, :self.modes4], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-4), x.size(-3), x.size(-2), x.size(-1)))
        return x

class FNO4d(nn.Module):
    def __init__(self, modes1, modes2, modes3, modes4, width):
        super(FNO4d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the first 10 timesteps + 4 locations (u(1, x, y, z), ..., u(10, x, y, z),  x, y, z, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, z=64, t=40, c=15)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, z=64, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.modes4 = modes4
        self.width = width
        self.padding = 6 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(7, self.width)
        # self.fc0 = nn.Linear(15, self.width)
        # input channel is 15: the solution of the first 10 timesteps + 4 locations (u(1, x, y, z), ..., u(10, x, y, z),  x, y, z, t)

        self.conv0 = SpectralConv4d(self.width, self.width, self.modes1, self.modes2, self.modes3, self.modes4)
        self.conv1 = SpectralConv4d(self.width, self.width, self.modes1, self.modes2, self.modes3, self.modes4)
        self.conv2 = SpectralConv4d(self.width, self.width, self.modes1, self.modes2, self.modes3, self.modes4)
        self.conv3 = SpectralConv4d(self.width, self.width, self.modes1, self.modes2, self.modes3, self.modes4)
        self.w0 = Conv4d(self.width, self.width, 1)
        self.w1 = Conv4d(self.width, self.width, 1)
        self.w2 = Conv4d(self.width, self.width, 1)
        self.w3 = Conv4d(self.width, self.width, 1)
        self.bn0 = BatchNorm4d(self.width)
        self.bn1 = BatchNorm4d(self.width)
        self.bn2 = BatchNorm4d(self.width)
        self.bn3 = BatchNorm4d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        # print('x shape', x.shape)
        grid = self.get_grid(x.shape, x.device)
        # print('grid shape', grid.shape)
        x = torch.cat((x, grid), dim=-1)
        # print('x shape', x.shape)

        x = self.fc0(x)
        x = x.permute(0, 5, 1, 2, 3, 4)
        x = F.pad(x, [0, self.padding]) # pad the domain if input is non-periodic

        del grid
        torch.cuda.empty_cache()

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2

        del x1, x2
        torch.cuda.empty_cache()

        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2

        del x1, x2  
        torch.cuda.empty_cache() 

        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2

        del x1, x2
        torch.cuda.empty_cache()

        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        del x1, x2
        torch.cuda.empty_cache()

        x = x[..., :-self.padding]
        x = x.permute(0, 2, 3, 4, 5, 1) # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        return x

    def get_grid(self, shape, device):

        batchsize, size_x, size_y, size_z, size_t, dim = shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]

        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, size_t, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1, 1).repeat([batchsize, size_x, 1, size_z, size_t, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1, 1).repeat([batchsize, size_x, size_y, 1, size_t, 1])
        gridt = torch.tensor(np.linspace(0, 1, size_t), dtype=torch.float)
        gridt = gridt.reshape(1, 1, 1, 1, size_t, 1).repeat([batchsize, size_x, size_y, size_z, 1, 1])

        return torch.cat((gridx, gridy, gridz, gridt), dim=-1).to(device)


################################################################
# configs
################################################################
if __name__ == "__main__":

    # Ajustar el valor de max_split_size_mb a 128 (puedes cambiar este valor según tus necesidades)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    modes = 6 # Bajo de 8 a 6 porque creo que no puede se mayor que nt+1
    width = 16

    # batch_size = 16 # mhg - Estaba a 10
    batch_size = 4
    batch_size2 = batch_size

    # epochs = 500
    epochs = 100 # mhg - Estaba a 50
    learning_rate = 0.001
    scheduler_step = 100
    scheduler_gamma = 0.5

    print('Epochs: ', epochs, ',learning rate: ', learning_rate, ',shceduler_step: ', scheduler_step, ',scheduler_gamma: ', scheduler_gamma)

    # runtime = np.zeros(2, )
    t1 = default_timer()


    sub = 1
    S = 64 // sub
    T_in = 1
    T = 6

    ################################################################
    # load data
    ################################################################


    LOCAL = False
    HULKA = False
    GOOGLE = False
    FRACTAL = False
    SAURON = True

    if LOCAL:
        V0_PATH = 'D:/TFG/HULKA-MONICA/DataGenerators/BullEyes/V0s/StationaryLDDMM/'
        EPDIFFV_PATH = 'D:/TFG/HULKA-MONICA/DataGenerators/BullEyes/V0s/EPDiffStationaryLDDMM/'
    elif HULKA:
        V0_PATH = '../../../DataGenerators/BullEyes/V0s/StationaryLDDMM/'
        EPDIFFV_PATH = '../../../DataGenerators/BullEyes/V0s/EPDiffStationaryLDDMM/'
    elif GOOGLE:
        V0_PATH = '/content/drive/MyDrive/TFG/2D/Data/BullEyes/StationaryLDDMM/'
        EPDIFFV_PATH = '/content/drive/MyDrive/TFG/2D/Data/BullEyes/EPDiffStationaryLDDMM/'
    elif FRACTAL:
        V0_PATH = '/FractalExt4/TFGs/fno/matlab/Metrics/OASIS_data_v0/'
        EPDIFFV_PATH = '/FractalExt4/TFGs/fno/matlab/Metrics/OASIS_data_v1/'
    elif SAURON:
        # V0_PATH = '/home/epdiff/python/3D/Data/OASIS_54_74_64/OASIS_data_v0/'
        # EPDIFFV_PATH = '/home/epdiff/python/3D/Data/OASIS_54_74_64/OASIS_data_v1/'

        # V0_PATH = '/home/epdiff/matlab/Metrics/OASIS_new_data_v0/'
        # EPDIFFV_PATH = '/home/epdiff/matlab/Metrics/OASIS_new_data_vt/'

        V0_PATH = '/home/epdiff/python/3D/Data/OASIS_54_74_64_nt5/OASIS_v0_nt5/'
        EPDIFFV_PATH = '/home/epdiff/python/3D/Data/OASIS_54_74_64_nt5/OASIS_vt_nt5/'
    
    v0_files = [f for f in os.listdir(V0_PATH) if f.startswith("OASIS") and f.endswith(".mat")]
    # print( v0_files )

    # mhg - Developing
    # v0_files = v0_files[0:100]
    # mhg - Developing

    from sklearn.model_selection import train_test_split

    test_v0_f = v0_files[394:]
    train_v0_f, val_v0_f = train_test_split( v0_files[:393], test_size=0.2, random_state=1234)

    ntrain = len(train_v0_f)
    nval = len(val_v0_f)
    ntest = len(test_v0_f)

    print('train: ', ntrain, ' val: ', nval, ' test: ', ntest)
    
    train_EPDiffv_f = [ "%s_EPDiff.mat" % f[:-4]  for f in train_v0_f ]
    val_EPDiffv_f = [ "%s_EPDiff.mat" % f[:-4] for f in val_v0_f ]
    test_EPDiffv_f = [ "%s_EPDiff.mat" % f[:-4] for f in test_v0_f ]

    train_v0 = []
    time_0 = time.time()
    for f in train_v0_f:
        data = scipy.io.loadmat(os.path.join(V0_PATH, f))    
        train_v0.append( data['v0'] )

    time_1 = time.time()
    print( 'v0 train loaded successfully in %f seconds' % (time_1 - time_0) )

    train_v = []
    for f in train_EPDiffv_f:
        data = scipy.io.loadmat(os.path.join(EPDIFFV_PATH, f))
        train_v.append(data['vt'])

    time_2 = time.time()
    print( 'vt train loaded successfully in %f seconds' % (time_2 - time_1) )

    val_v0 = []
    for f in val_v0_f:
        data = scipy.io.loadmat(os.path.join(V0_PATH, f))
        val_v0.append( data['v0'] )

    time_3 = time.time()
    print( 'v0 val loaded successfully in %f seconds' % (time_3 - time_2) )

    val_v = []
    for f in val_EPDiffv_f:
        data = scipy.io.loadmat(os.path.join(EPDIFFV_PATH, f))
        val_v.append(data['vt'])

    time_4 = time.time()
    print( 'vt val loaded successfully in %f seconds' % (time_4 - time_3) )

    test_v0 = []
    for f in test_v0_f:
        data = scipy.io.loadmat(os.path.join(V0_PATH, f))
        test_v0.append(data['v0'])

    time_5 = time.time()
    print( 'v0 test loaded successfully in %f seconds' % (time_5 - time_4) )

    test_v = []
    for f in test_EPDiffv_f:
        data = scipy.io.loadmat(os.path.join(EPDIFFV_PATH, f))
        test_v.append(data['vt'])

    time_6 = time.time()
    print( 'vt test loaded successfully in %f seconds' % (time_6 - time_5) )

    train_v0 = torch.tensor(np.array(train_v0))
    train_v = torch.tensor(np.array(train_v))

    val_v0 = torch.tensor(np.array(val_v0))
    val_v = torch.tensor(np.array(val_v))

    test_v0 = torch.tensor(np.array(test_v0))
    test_v = torch.tensor(np.array(test_v))


    longitud = len(train_v)
    # Para cada elemento de train_v muestra el mayor y menor
    for i in range(0, longitud):
        if(i >= len(train_v)):
            break
        # print("Maximo: ", train_v[i].max(), "Minimo: ", train_v[i].min())
        if train_v[i].max() > 1 and train_v[i].min() < -1:
            # Eliminar del conjunto de entrenamiento
            train_v0 = torch.cat((train_v0[:i], train_v0[i+1:]), 0)
            train_v = torch.cat((train_v[:i], train_v[i+1:]), 0)
            print("Eliminado")
            longitud -= 1
            i -= 1
        


    print("Longitud: ", len(train_v))


    ################################################################
    # normalize data
    ################################################################

    GAUSSNORM = False
    NORM_1_1 = True
    NORM_0_1 = False


    if GAUSSNORM:
        print('Normalizing data with Gaussian normalizer')
    
        normalizer_v0 = UnitGaussianNormalizer(train_v0)
        torch.save(normalizer_v0, 'OASIS_FNO3D_GaussNormalizer_v0.pt')
        train_v0 = normalizer_v0.encode(train_v0)
        val_v0 = normalizer_v0.encode(val_v0)
        test_v0 = normalizer_v0.encode(test_v0)

        normalizer_vt = UnitGaussianNormalizer(train_v)
        torch.save(normalizer_vt, 'OASIS_FNO3D_GaussNormalizer_vt.pt')
        train_v = normalizer_vt.encode(train_v)
        val_v = normalizer_vt.encode(val_v)
        test_v = normalizer_vt.encode(test_v)

    elif NORM_1_1:
        print('Normalizing data in range [-1, 1]')

        min_v0 = train_v0.min()
        max_v0 = train_v0.max()
        print('min_v0: ', min_v0, ', max_v0: ', max_v0)

        train_v0 = 2.0 * (train_v0 - min_v0) / (max_v0 - min_v0) - 1.0
        val_v0 = 2.0 * (val_v0 - min_v0) / (max_v0 - min_v0) - 1.0
        test_v0 = 2.0 * (test_v0 - min_v0) / (max_v0 - min_v0) - 1.0

        min_v = train_v.min()
        max_v = train_v.max()

        print('min_v: ', min_v, ', max_v: ', max_v)

        train_v = 2.0 * (train_v - min_v) / (max_v - min_v) - 1.0
        val_v = 2.0 * (val_v - min_v) / (max_v - min_v) - 1.0
        test_v = 2.0 * (test_v - min_v) / (max_v - min_v) - 1.0

    elif NORM_0_1:
        print('Normalizing data in range [0, 1]')

        min_v0 = train_v0.min()
        max_v0 = train_v0.max()
        print('min_v0: ', min_v0, ', max_v0: ', max_v0)

        train_v0 = (train_v0 - min_v0) / (max_v0 - min_v0)
        val_v0 = (val_v0 - min_v0) / (max_v0 - min_v0)
        test_v0 = (test_v0 - min_v0) / (max_v0 - min_v0)

        min_v = train_v.min()
        max_v = train_v.max()

        print('min_v: ', min_v, ', max_v: ', max_v)

        # Aplanar el array
        flattened_v = train_v.flatten()

        # Obtener los 10 valores menores usando np.partition
        min_10_values = np.partition(flattened_v, 10)[:10]
        min_10_values = np.sort(min_10_values)  # Ordenar los 10 valores menores obtenidos

        # Obtener los 10 valores mayores usando np.partition
        max_10_values = np.partition(flattened_v, -10)[-10:]
        max_10_values = np.sort(max_10_values)  # Ordenar los 10 valores mayores obtenidos

        print("Primeros 10 valores menores:", min_10_values)
        print("Primeros 10 valores mayores:", max_10_values)


        min_v = val_v.min()
        max_v = val_v.max()

        print('min_v: ', min_v, ', max_v: ', max_v)

        min_v = test_v.min()
        max_v = test_v.max()

        print('min_v: ', min_v, ', max_v: ', max_v)

        exit(0)

        train_v = (train_v - min_v) / (max_v - min_v)
        val_v = (val_v - min_v) / (max_v - min_v)
        test_v = (test_v - min_v) / (max_v - min_v) 

    else:
        print('No normalization')

    class DatasetArray(torch.utils.data.Dataset):
        'Characterizes a dataset for PyTorch'

        def __init__(self, v0, v):
            'Initialization'
            self.v0 = v0
            self.v = v

        def __len__(self):
            'Denotes the total number of samples'
            return len(self.v0)

        def __getitem__(self, index):
            'Generates one sample of data'
            # Select sample
            X = self.v0[index]
            XX = self.v[index]

            X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1, 3).repeat([1,1,1,T,1])

            return X, XX

    train_set = DatasetArray( train_v0, train_v )
    val_set = DatasetArray( val_v0, val_v )
    test_set = DatasetArray( test_v0, test_v )

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # val_loader = torch.utils.data.DataLoader(val_set, batch_size=len(val_v0), shuffle=False)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size = len(test_v0), shuffle = False )

    print( len(val_v0), len(test_v0) )

    val_batch_size = 1
    test_batch_size = 1

    # val_batch_size = len(val_v0)
    # test_batch_size = len(test_v0)

    val_loader = torch.utils.data.DataLoader(val_set, batch_size=val_batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle = False )

    t2 = default_timer()

    print('preprocessing finished, time used:', t2-t1)
    device = torch.device('cuda')

    ################################################################
    # training and evaluation
    ################################################################
    print('Creating model...')
    model = FNO4d(modes, modes, modes, modes, width).cuda()

    print('Count params: ', count_params(model))
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    myloss = LpLoss(size_average=False)

    best_model_state = None
    BEST_val_loss = 1000
    itBest = 0

    
    print('Start training...')
    # get_gpu_info()
    for ep in range(epochs):

        model.train()

        t1 = default_timer()

        train_l2_step = 0
        val_l2_step = 0
        train_l2_full = 0

        for x, y in train_loader:

            x, y = x.cuda(), y.cuda()


            out = model(x)

            out = out.permute(0,1,2,3,5,4)

            loss = myloss(out, y)

            train_l2_step += loss.item()

            optimizer.zero_grad()
            # mse.backward()
            loss.backward()
            optimizer.step()


        # get_gpu_info()
        # Eliminar variables no referenciadas de la memoria GPU
        del loss, x, y, out
        torch.cuda.empty_cache()
        # print('ELIMINADO BASURA')
        # get_gpu_info()


        # mhg - Select the best model in the validation set from the last 50 epochs
        if ep > epochs - 50:

            model.eval()

            with (torch.no_grad()):

                val_l2_step = 0

                for x, y in val_loader:

                    x, y = x.cuda(), y.cuda()

                    # get_gpu_info()

                    out = model(x)
                    out = out.permute(0,1,2,3,5,4)


                    val_loss = myloss(out.reshape(out.shape[0], -1), y.reshape(out.shape[0], -1))

                    val_l2_step += val_loss.item()

            if val_l2_step < BEST_val_loss:
                print("Before %f - After %f \n" % (BEST_val_loss, val_l2_step))
                BEST_val_loss = val_l2_step
                best_model_state = model.state_dict()

            del x, y, out, val_loss
            torch.cuda.empty_cache()
            # print('ELIMINADO BASURA VALIDACION')
            # get_gpu_info()

        t2 = default_timer()
        scheduler.step()

        if ep > epochs - 50:
            print(ep, t2 - t1, train_l2_step / ntrain, val_l2_step / nval)
        else:
            print(ep, t2 - t1, train_l2_step / ntrain )

        # Eliminar variables no referenciadas de la memoria GPU
        # del loss, x, y, out
        # if ep > epochs - 50:
        #     del val_loss
        # torch.cuda.empty_cache()


    model.load_state_dict(best_model_state)

    if GAUSSNORM:
        torch.save(model, 'OASIS_FNO4D_GaussNorm_model_A.pt')
    elif NORM_1_1:
        torch.save(model, 'OASIS_FNO3D_T_1_1_nt5_model_R.pt')
    elif NORM_0_1:
        torch.save(model, 'OASIS_FNO3D_T_0_1_nt5_model_R.pt')
    else:
        torch.save(model, 'OASIS_FNO3D_T_nt5_model.pt' )
    

    # model = torch.load('OASIS_FNO4D_model_A.pt')

    test_l2_step = 0
    test_l2_full = 0

    with torch.no_grad():

        model.eval()

        for xx, yy in test_loader:

            test_loss = 0

            if bigGPU:

                xx = xx.to(device)  # v0
                yy = yy.to(device)  # v

                vv = model(xx)

            else:

                vv = model.cpu()(xx.cpu())
                model = model.to(device)

            # vv = vv.view(vv.shape[0], 96, 96, 2, T)
            vv = vv.permute(0,1,2,3,5,4)

            print(vv.shape, yy.shape)

            test_loss = myloss(vv.reshape(vv.shape[0], -1), yy.reshape(vv.shape[0], -1))

            """
            for t in range(0, T, step):
                y = yy[..., t:t + step]
                im = model(xx)
                loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

                xx = torch.cat((xx[..., step:], im), dim=-1)

            """

            test_l2_step += test_loss.item()
            # test_l2_full += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

            # Liberar la memoria no referenciada
            del xx, yy, vv
            torch.cuda.empty_cache()

        print('Final results train - val - test')
        print(train_l2_step / ntrain, BEST_val_loss / nval, test_l2_step / ntest)
        print(test_l2_step / ntest)
