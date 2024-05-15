#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Benedikt Barthel Sorensen: bbarthel@mit.edu
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as kb
import tensorflow.keras.losses
#import tensorflow_probability as tfp
import tensorflow.signal as tfs
import tensorflow.math as tfm
import scipy.interpolate as sci
#%%
class loss_params:
    def __init__(self, kd, delta, Ny, yy):
        self.kd = kd
        self.delta = delta
        self.Ny = Ny
        self.dy = 2*np.pi/float(Ny)
        self.yy = yy
        
        
        
class loss_params_2D:
    def __init__(self, kd, delta, Ny, Nx, beta, rdrag):
        self.kd = kd
        self.delta = delta
        self.beta  = beta
        self.rdrag = rdrag
        self.Ny = Ny
        self.Nx = Nx
        self.dy = 2.0*np.pi/float(Ny)
        self.dx = 2.0*np.pi/float(Nx)
     
  
        
        
class loss_params_2DV:
    def __init__(self, kd, delta, Ny, Nx, beta, rdrag, latent_dim):
        self.kd = kd
        self.delta = delta
        self.beta  = beta
        self.rdrag = rdrag
        self.Ny = Ny
        self.Nx = Nx
        self.dy = 2.0*np.pi/float(Ny)
        self.dx = 2.0*np.pi/float(Nx)   
        self.latent_dim = latent_dim
        
        
        
        
        
        
        
        
        
        
        
        
        
       