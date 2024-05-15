#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Benedikt Barthel Sorensen <bbarthel@mit.edu>
"""

#from DNS_Class import dns
#from Test_Cases import Define_QG_Model

import sys # Imported to read input from command line.
import numpy as np
import scipy.io as sio
import scipy.signal as scs
import os
import tensorflow as tf
import keras
from keras import layers


from keras import backend as K
import tensorflow.keras.backend as kb
from keras.models import Sequential, Model
from keras.layers import Input, LSTM, RepeatVector
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.optimizers import SGD, RMSprop, Adam
#from keras import objectives


from loss_params_qg import *
from loss_funs_qg import *



def get_model(model_type,used_features,loss_data):
    
    
    if model_type == 'LSTM' or model_type == 'LSTMgeo':
        print("Standard LSTM Network")
        network = get_lstm(used_features,loss_data)
    
    if model_type == 'VLSTM7':
        print("Variational LSTM Network #7")
        network = get_vlstm7(used_features,loss_data)
        
    if model_type == 'STORN1':
        print("Variational STORN Network #1")
        network = get_storn1(used_features,loss_data)
   
    
        
        
        
    if model_type == 'VRNN1':
         print("Variational RNN Network #1")
         network = get_vrnn1(used_features,loss_data)
    
         
         
    return network



def get_lstm(used_features,loss_data):
    
    
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(None,used_features)))
    model.add(tf.keras.layers.Dense(60, activation='tanh'))
    model.add(tf.keras.layers.LSTM(input_shape=(None,60),units=60, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.00, recurrent_dropout=0.00, implementation=1, return_sequences=True, return_state=False, go_backwards=False, stateful=False, unroll=False))
    model.add(tf.keras.layers.Dense(used_features, activation='linear'))
    model.compile(loss=param_loss_2D(alpha=loss_data), optimizer='adam' )



    return model




# VAE-RNN with lambda =  10^-4 loss
def get_vaernn(used_features,loss_data):
    
    features = used_features
    inter_dim = 60
    latent_dim = 60
    
    def sampling(args):
        z_mean, z_sigma = args
        batch_size = tf.shape(z_mean)[0] # <================
        time_hist = tf.shape(z_mean)[1] # <================
        epsilon = kb.random_normal(shape=(batch_size,time_hist, latent_dim), mean=0., stddev=1.)
        return z_mean + (z_sigma)*epsilon
   
    

   
    # timesteps, features
    x_in = tf.keras.layers.Input(shape= (None, features),name = 'Input') 
    
    # Dense Layer
    h = tf.keras.layers.Dense(inter_dim, activation='tanh', name="Encoder1")(x_in)
    
    #z_layer
    z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean", activation='linear')(h)
    z_sigma = tf.keras.layers.Dense(latent_dim, name="z_var", activation='softplus',kernel_initializer='zeros')(h)
    #z = tf.keras.layers.Dense(latent_dim)(h)
    z = tf.keras.layers.Lambda(sampling, name="Sample")([z_mean, z_sigma])
    
    # Post Z layer LSTM
    z = tf.keras.layers.LSTM(input_shape=(None,60),units=60, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.00, recurrent_dropout=0.00, implementation=1, return_sequences=True, return_state=False, go_backwards=False, stateful=False, unroll=False, name="LSTM2")(z)
    
    
    # Reconstruction decoder
    #decoder1 = tf.keras.layers.TimeDistributed(Dense(features))(z)
    y_pred = tf.keras.layers.Dense(used_features, activation='linear', name="Decoder1")(z)
    
    # Concatonate
    output = tf.keras.layers.Concatenate(axis=-1, name="Concatenate")([y_pred, z_mean, z_sigma])
    
    # Create Model
    model = tf.keras.Model(inputs=[x_in],outputs=[output],)
    model.compile(loss=vae_loss_2D(alpha=loss_data), optimizer='adam' )


    return model



# STORN with lambda =  10^-4 loss
def get_storn1(used_features,loss_data):
    
    features = used_features
    inter_dim = 60
    latent_dim = 60
    
    def sampling(args):
        z_mean, z_sigma = args
        batch_size = tf.shape(z_mean)[0] # <================
        time_hist = tf.shape(z_mean)[1] # <================
        epsilon = kb.random_normal(shape=(batch_size,time_hist, latent_dim), mean=0., stddev=1.)
        return z_mean + (z_sigma)*epsilon
   
    

   
    # timesteps, features
    x_in = tf.keras.layers.Input(shape= (None, features),name = 'Input') 
    
    # Dense Layer
    h = tf.keras.layers.Dense(inter_dim, activation='tanh', name="Encoder1")(x_in)
    
    #z_layer
    z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean", activation='linear')(h)
    z_sigma = tf.keras.layers.Dense(latent_dim, name="z_var", activation='softplus',kernel_initializer='zeros')(h)
    z = tf.keras.layers.Lambda(sampling, name="Sample")([z_mean, z_sigma])
    
    # Concatonate
    phi =  tf.keras.layers.Concatenate(axis=-1, name="Concatenate1")([h,z])
    
    # Post Z layer LSTM
    d = tf.keras.layers.LSTM(input_shape=(None,latent_dim+inter_dim),units=inter_dim, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.00, recurrent_dropout=0.00, implementation=1, return_sequences=True, return_state=False, go_backwards=False, stateful=False, unroll=False, name="LSTM2")(phi)
    
    
    # Reconstruction decoder
    #decoder1 = tf.keras.layers.TimeDistributed(Dense(features))(z)
    y_pred = tf.keras.layers.Dense(used_features, activation='linear', name="Decoder1")(d)
    
    # Concatonate
    output = tf.keras.layers.Concatenate(axis=-1, name="Concatenate2")([y_pred, z_mean, z_sigma])
    
    # Create Model
    model = tf.keras.Model(inputs=[x_in],outputs=[output],)
    model.compile(loss=vae_loss_2D(alpha=loss_data), optimizer='adam' )


    return model




# VRNN-I with lambda =  10^-4 loss
def get_vrnn1(used_features,loss_data):
    
    features = used_features
    inter_dim = 60
    latent_dim = 60
    
    def sampling(args):
        z_mean, z_sigma = args
        batch_size = tf.shape(z_mean)[0] # <================
        time_hist = tf.shape(z_mean)[1] # <================
        epsilon = kb.random_normal(shape=(batch_size,time_hist, latent_dim), mean=0., stddev=1.)
        return z_mean + (z_sigma)*epsilon
   
    

   
    # timesteps, features
    x_in = tf.keras.layers.Input(shape= (None, features),name = 'Input') 
    
    # Dense Layer
    h = tf.keras.layers.Dense(inter_dim, activation='tanh', name="Encoder1")(x_in)
    
    

    #z_layer
    z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean", activation='linear')(h)
    z_sigma = tf.keras.layers.Dense(latent_dim, name="z_var", activation='softplus',kernel_initializer='zeros')(h)
    z = tf.keras.layers.Lambda(sampling, name="Sample")([z_mean, z_sigma])
    
    # Concatonate
    phi =  tf.keras.layers.Concatenate(axis=-1, name="Concatenate1")([h,z])
    
    # Post Z layer LSTM
    d = tf.keras.layers.LSTM(input_shape=(None,latent_dim+inter_dim),units=inter_dim, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.00, recurrent_dropout=0.00, implementation=1, return_sequences=True, return_state=False, go_backwards=False, stateful=False, unroll=False, name="LSTM2")(phi)
    
    # Concatonate
    zd =  tf.keras.layers.Concatenate(axis=-1, name="Concatenate0")([z,d])
    
    
    # Reconstruction decoder
    #decoder1 = tf.keras.layers.TimeDistributed(Dense(features))(z)
    y_pred = tf.keras.layers.Dense(used_features, activation='linear', name="Decoder1")(zd)
    
    # Concatonate
    output = tf.keras.layers.Concatenate(axis=-1, name="Concatenate2")([y_pred, z_mean, z_sigma])
    
    # Create Model
    model = tf.keras.Model(inputs=[x_in],outputs=[output],)
    model.compile(loss=vae_loss_2D(alpha=loss_data), optimizer='adam' )


    return model

















