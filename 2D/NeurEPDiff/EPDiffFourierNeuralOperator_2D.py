"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Navier-Stokes equation discussed in Section 5.3 in the [paper](https://arxiv.org/pdf/2010.08895.pdf),
which uses a recurrent structure to propagates in time.
"""

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


bigGPU = True # False # GeForce 1080 ti, 11 GBs

################################################################
# fourier layer
################################################################

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(4, self.width)
        # self.fc0 = nn.Linear(12, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        # self.fc2 = nn.Linear(128, 1)
        self.fc2 = nn.Linear(128, 2*31)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        # print( 'grid.shape', grid.shape )
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        # print( 'x.shape 1 capa', x.shape )
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        # print( 'x.shape 2 capa', x.shape )
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        # print( 'x.shape 3 capa', x.shape )
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        # print( 'x.shape 4 capa', x.shape )
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        # print( 'x.shape before permute', x.shape )
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        # print( 'x.shape', x.shape )
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, dim = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


################################################################
# configs
################################################################
if __name__ == "__main__":
    # Ajustar el valor de max_split_size_mb a 128 (puedes cambiar este valor según tus necesidades)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'

    ntrain = 800
    nval = 200
    ntest = 200

    modes = 12
    width = 20

    batch_size = 16 # mhg - Estaba a 20
    batch_size2 = batch_size

    # epochs = 500
    epochs = 100
    learning_rate = 0.001
    scheduler_step = 100
    scheduler_gamma = 0.5

    print('Epochs: ', epochs, ',learning rate: ', learning_rate, ',shceduler_step: ', scheduler_step, ',scheduler_gamma: ', scheduler_gamma)


    path = 'ns_fourier_2d_rnn_V10000_T20_N' + str(ntrain) + '_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width)
    path_model = 'model/' + path
    path_train_err = 'results/' + path + 'train.txt'
    path_test_err = 'results/' + path + 'test.txt'
    path_image = 'image/' + path


    sub = 1
    S = 64
    T_in = 10
    T = 31
    step = 1

    ################################################################
    # load data
    ################################################################

    model = FNO2d(modes, modes, width).cuda()
    # model = FNO2d(modes, modes, width)

    print('Model parameters: ', count_params(model))

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
        V0_PATH = '/FractalExt4/TFGs/fno/2D/Data/BullEyes/V0s/StationaryLDDMM/'
        EPDIFFV_PATH = '/FractalExt4/TFGs/fno/2D/Data/BullEyes/V0s/EPDiffStationaryLDDMM/'
    elif SAURON:
        V0_PATH = '/home/epdiff/python/2D/Data/StationaryLDDMM/'
        EPDIFFV_PATH = '/home/epdiff/python/2D/Data/EPDiffStationaryLDDMM/'

        

    v0_files = [f for f in os.listdir(V0_PATH) if f.startswith("Bull") and f.endswith(".mat")]

    print( v0_files )
    print( 'Total files: ', len(v0_files) )


    from sklearn.model_selection import train_test_split

    train_v0_f, test_v0_f = train_test_split(v0_files, test_size=0.2, random_state=1234)
    train_v0_f, val_v0_f = train_test_split( train_v0_f, test_size=0.2, random_state=12334)

    train_EPDiffv_f = [ "BullEye_EPDiffv_%s" % f[11:] for f in train_v0_f ]
    val_EPDiffv_f = [ "BullEye_EPDiffv_%s" % f[11:] for f in val_v0_f ]
    test_EPDiffv_f = [ "BullEye_EPDiffv_%s" % f[11:] for f in test_v0_f ]

    time_0 = time.time()
    train_v0 = []

    for f in train_v0_f:
        data = scipy.io.loadmat(os.path.join(V0_PATH, f))
        train_v0.append( data['v0'] )

    time_1 = time.time()
    print( 'train v0 loaded successfully in %f sec' % (time_1 - time_0) )

    train_v = []
    for f in train_EPDiffv_f:
        data = scipy.io.loadmat(os.path.join(EPDIFFV_PATH, f))
        train_v.append(data['v'])

    time_2 = time.time()
    print( 'train v loaded successfully in %f sec' % (time_2 - time_1) )

    val_v0 = []
    for f in val_v0_f:
        data = scipy.io.loadmat(os.path.join(V0_PATH, f))
        val_v0.append( data['v0'] )

    time_3 = time.time()
    print( 'val v0 loaded successfully in %f sec' % (time_3 - time_2) )

    val_v = []
    for f in val_EPDiffv_f:
        data = scipy.io.loadmat(os.path.join(EPDIFFV_PATH, f))
        val_v.append(data['v'])

    time_4 = time.time()
    print( 'val v loaded successfully in %f sec' % (time_4 - time_3) )

    test_v0 = []
    for f in test_v0_f:
        data = scipy.io.loadmat(os.path.join(V0_PATH, f))
        test_v0.append(data['v0'])

    time_5 = time.time()
    print( 'test v0 loaded successfully in %f sec' % (time_5 - time_4) )

    test_v = []
    for f in test_EPDiffv_f:
        data = scipy.io.loadmat(os.path.join(EPDIFFV_PATH, f))
        test_v.append(data['v'])

    time_6 = time.time()
    print( 'test v loaded successfully in %f sec' % (time_6 - time_5) )

    train_v0 = torch.tensor(np.array(train_v0))
    train_v = torch.tensor(np.array(train_v))

    val_v0 = torch.tensor(np.array(val_v0))
    val_v = torch.tensor(np.array(val_v))

    test_v0 = torch.tensor(np.array(test_v0))
    test_v = torch.tensor(np.array(test_v))


    ################################################################
    # normalize data
    ################################################################

    GAUSSNORM = False
    NORM_1_1 = False
    NORM_0_1 = False


    if GAUSSNORM:
        print('Normalizing data with Gaussian normalizer')
    
        normalizer_v0 = UnitGaussianNormalizer(train_v0)
        torch.save(normalizer_v0, 'BullEye_FNO2D_GaussNormalizer.pt')
        train_v0 = normalizer_v0.encode(train_v0)
        val_v0 = normalizer_v0.encode(val_v0)
        test_v0 = normalizer_v0.encode(test_v0)

        normalizer_v = UnitGaussianNormalizer(train_v)
        train_v = normalizer_v.encode(train_v)
        val_v = normalizer_v.encode(val_v)
        test_v = normalizer_v.encode(test_v)

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

        train_v = (train_v - min_v) / (max_v - min_v)
        val_v = (val_v - min_v) / (max_v - min_v)
        test_v = (test_v - min_v) / (max_v - min_v) 

    else:
        print('No normalization')


    ################################################################
    # create data loaders
    ################################################################
    
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
            
            # COMENTADO POR ALVARO
            # X = torch.tensor(X)
            # XX = torch.tensor(XX)

            # X = X.unsqueeze(0)
            # X = X.flatten()

            # XX = XX.unsqueeze(0)
            # XX = XX.flatten()

            return X, XX

    train_set = DatasetArray( train_v0, train_v )
    val_set = DatasetArray( val_v0, val_v )
    test_set = DatasetArray( test_v0, test_v )

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    print('len(val_v0): ', len(val_v0), 'len(test_v0); ', len(test_v0) )

    val_batch_size = 32
    test_batch_size = 32


    val_loader = torch.utils.data.DataLoader(val_set, batch_size=val_batch_size, shuffle=False)

    test_batch_size = len(test_v0)  # Todavia no funciona en trocitos la parte de salvar
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle = False )

    ################################################################
    # training and evaluation
    ################################################################

    print('Count params: ', count_params(model))
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    myloss = LpLoss(size_average=False)

    best_model_state = None
    BEST_val_loss = 1000
    itBest = 0

    time_7 = time.time()
    for ep in range(epochs):

        model.train()

        t1 = default_timer()

        train_l2_step = 0
        val_l2_step = 0
        train_l2_full = 0

        for xx, yy in train_loader:
            
            loss = 0
            
            xx = xx.to(device) # v0
            yy = yy.to(device) # v

            v = model(xx)

            # print( 'v.shape', v.shape )
            # exit(0)

            loss = myloss(v.reshape(batch_size, -1), yy.reshape(batch_size, -1))

            # print( loss )

            """
            Aqui se cogen como input lo estimado de [0 a tal] y despues se estima en el t+1 y se incorpora en la prediccion
            y asi sucesivamente...
            for t in range(0, T, step): ???
                y = yy[..., t:t + step]
                im = model(xx)
                loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

                xx = torch.cat((xx[..., step:], im), dim=-1)
            """

            train_l2_step += loss.item()
            # l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
            # train_l2_full += l2_full.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # mhg - Select the best model in the validation set from the last 50 epochs
        if ep > epochs - 50:

            model.eval()

            with torch.no_grad():

                val_l2_step = 0

                for xx, yy in val_loader:

                    if bigGPU:

                        xx = xx.to(device)  # v0
                        yy = yy.to(device)  # v

                        vv = model(xx)

                    else:

                        vv = model.cpu()(xx.cpu())

                        model = model.to(device)

                    val_loss = myloss(vv.reshape(batch_size, -1), yy.reshape(batch_size, -1))

                    val_l2_step += val_loss.item() # mhg - Se puede pasar varias veces por el bucle si no se coge el val/test set de tamaño maximo

                # print( "mhg?")
                # print(BEST_val_loss, val_loss_step, val_l2_step / nval)

                if val_l2_step < BEST_val_loss:
                    print( "Before %f - After %f \n" % (BEST_val_loss, val_l2_step) )
                    BEST_val_loss = val_l2_step
                    best_model_state = model.state_dict()
                    # print( best_model_state['fc2.bias'])


        t2 = default_timer()
        scheduler.step()
        # print(ep, t2 - t1, train_l2_step / ntrain / (T / step), train_l2_full / ntrain, test_l2_step / ntest / (T / step),
        #      test_l2_full / ntest)

        if ep > epochs - 50:
            print(ep, t2 - t1, train_l2_step / ntrain, val_l2_step / nval)
        else:
            print(ep, t2 - t1, train_l2_step / ntrain )

        # Eliminar variables no referenciadas de la memoria GPU
        del loss, v, xx, yy
        torch.cuda.empty_cache()
        # Liberar la memoria no referenciada
        torch.cuda.memory_reserved(0)
        torch.cuda.memory_allocated(0)

    time_8 = time.time()
    print( 'Training done in %f sec' % (time_8 - time_7) )

    model.load_state_dict(best_model_state)

    test_l2_step = 0
    test_l2_full = 0

    print( 'Starting test' )
    with torch.no_grad():

        model.eval()

        for xx, yy in test_loader:

            test_loss = 0

            if bigGPU:
                xx = xx.to(device)  # v0
                vv = model(xx)

                del xx 
                torch.cuda.empty_cache()

                yy = yy.to(device)  # v
        
                # vv = model(xx)

            else:

                vv = model.cpu()(xx.cpu())
                model = model.to(device)

            test_loss = myloss(vv.reshape(batch_size, -1), yy.reshape(batch_size, -1))

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
            del yy, vv
            torch.cuda.empty_cache()
            torch.cuda.memory_reserved(0)
            torch.cuda.memory_allocated(0)

        time_9 = time.time()
        print( 'Test done in %f sec' % (time_9 - time_8) )

        print( 'Final results train - val - test' )
        print(train_l2_step / ntrain, val_l2_step / nval, test_l2_step / ntest )
        print(train_l2_step / ntrain, BEST_val_loss / nval, test_l2_step / ntest)
        

    if GAUSSNORM:
        torch.save(model, 'BullEye_FNO2D_GaussNorm_model.pt')
    elif NORM_1_1:
        torch.save(model, 'BullEye_FNO2D_1_1_model.pt')
    elif NORM_0_1:
        torch.save(model, 'BullEye_FNO2D_0_1_model.pt')
    else:
        torch.save(model, 'BullEye_FNO2D_model.pt' )

    

    # mhg - BEGIN Descomentar para guardar

    # with torch.no_grad():

    #     j = 0

    #     for xx, yy in test_loader:

    #         if bigGPU:

    #             xx = xx.to(device)
    #             yy = yy.to(device)

    #             vv = model(xx)

    #         else:

    #             vv = model.cpu()(xx.cpu())

    #             model = model.to(device)

    #         # scipy.io.savemat( 'BullEye_FNO2D_%s' % test_EPDiffv_f[i][16:], {'v': vv.numpy()} )

    #     if use_encoding:
    #         vv = normalizer_v.decode(vv.cpu())
    #     else:
    #         vv = vv.cpu()

    #     for i in range(vv.shape[0]):
    #         scipy.io.savemat('BullEye_FNO2D_%s' % test_EPDiffv_f[j][16:], {'v': np.squeeze(vv[i, :, :, :, :].numpy())})
    #         j = j + 1

    #     print("Writen %d minifiles" % vv.shape[0])

    #     print("Written %d files" % j)

    #     """"
    #     ... Usado antes para escribir todo de golpe ...
    #     for i in range(len(test_v0_f)):
    #         scipy.io.savemat('BullEye_FNO2D_%s' % test_EPDiffv_f[i][16:], {'v': np.squeeze( vv[i, :, :, :, :].numpy() )}) # Esto funciona si puedo hacer la inferencia una sola vez
    #         # i = i + 1 # Esto estaba mal, afortunadamente no parece haber influido
    #     """

    # mhg - END Descomentar para guardar

    # torch.save(model, path_model)

    # ruta_modelo = 'state_dict_model.pt'
    # torch.save(model.state_dict(), ruta_modelo)
    # print('Modelo guardado en %s' % ruta_modelo)

    # pred = torch.zeros(test_u.shape)
    # index = 0
    # test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)
    # with torch.no_grad():
    #     for x, y in test_loader:
    #         test_l2 = 0;
    #         x, y = x.cuda(), y.cuda()
    #
    #         out = model(x)
    #         out = y_normalizer.decode(out)
    #         pred[index] = out
    #
    #         test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
    #         print(index, test_l2)
    #         index = index + 1

    # scipy.io.savemat('pred/'+path+'.mat', mdict={'pred': pred.cpu().numpy()})
