#!/usr/bin/env python4
# -*- coding: utf-8 -*-
"""
@author: Benedikt Barthel Sorensen <bbarthel@mit.edu>
"""

#from DNS_Class import dns
#from Test_Cases import Define_QG_Model

import sys # Imported to read input from command line.
import numpy as np
import scipy.io as sio
import os


print('Importing Tensorflow... ', end='', flush=True)
import tensorflow as tf
print('Done!')
print('Importing Keras... ', end='', flush=True)
#from tensorflow import keras
print('Done!')

from ml_funs_qg import *

if __name__ == "__main__":



    user_input = []
    # Train Models
    user_input.append(sys.argv[0])
    user_input.append(sys.argv[1]) # model_type
    user_input.append(sys.argv[2]) # T
    user_input.append(sys.argv[3]) # Nepochs
    user_input.append(sys.argv[4]) # Nens

    # User Inputs
    model_type =user_input[1]
    T = int(user_input[2])
    Nepochs = int(user_input[3])
    Nens = int(user_input[4])

    # Data Settings
    basis_type1 ="rnudge16.0"
    case_name_train = "QG_Data_beta2.0_rdrag0.1_realization1_24x24"
    case_name_test = "QG_Data_beta2.0_rdrag0.1_realization2_24x24"

    # LSTM Settings
    dt_ml = 0.1


    #%% Test
    ML_Test_Ensemble(model_type,case_name_test,Nens,Nepochs,dt_ml,basis_type1,compute_pdf,save_full_field,T)
    
       
    exit()