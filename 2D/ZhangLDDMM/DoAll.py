import numpy as np
import scipy.io
from NIREPZhangGradientDescentMCCLDDMM_2d import NIREPZhangGradientDescentMCCLDDMM_2d
import os
import torch

import sys
sys.path.append('/content/drive/MyDrive/TFG/2D/NeurEPDiff')
sys.path.append(os.path.abspath('../NeurEPDiff'))
from EPDiffFourierNeuralOperator_2D import FNO2d
from EPDiffFourierNeuralOperator_2D import SpectralConv2d_fast

from EPDiffFourierNeuralOperator_2D_time import FNO3d
from EPDiffFourierNeuralOperator_2D_time import SpectralConv3d

reg_type = 'CN'
opt_type = 'GradientDescent'

# --------------------------------------------------- CONTROLLER -------------------------------------------------------

# Select the model:
#       0 -> ODESOLVER
#       1 -> FNO2D
#       2 -> FNO2D_time (FNO2D with convolution in the temporal space)
solver = 1


# ----------------------------------------------------------------------------------------------------------------------

if solver == 0:
    path_model = ''
elif solver == 1:
    path_model = '/home/epdiff/python/2D/NeurEPDiff/BullEye_FNO2D_model.pt'
elif solver == 2:
    path_model = '/home/epdiff/python/2D/NeurEPDiff/BullEye_FNO2D_T_model.pt'

for patient in range(2, 3):
    name = 'NIREPZhangMCC_{}_{}_{}.mat'.format(reg_type, opt_type, patient)
    vname = 'V_NIREPZhangMCC_{}_{}_{}.mat'.format(reg_type, opt_type, patient)
    
    print(name)
    
    try:
        print('Diffeo start')
        v0, history, history_all, Jacs, time, time_all = NIREPZhangGradientDescentMCCLDDMM_2d(patient, patient + 1, solver, path_model)
        
        # Guardar los resultados 
        scipy.io.savemat(name, {'history': history, 'history_all': history_all, 'Jacs': Jacs, 'time': time, 'time_all': time_all})
        scipy.io.savemat(vname, {'v0': v0.cpu().detach().numpy()})
        
    except Exception as e:
        print(f'Error in diffeo: {e}')
    
    print('Diffeo done')


