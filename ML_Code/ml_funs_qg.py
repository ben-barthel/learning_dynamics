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
import h5py
import hdf5storage


print('Importing Tensorflow... ', end='', flush=True)
import tensorflow as tf
print('Done!')
print('Importing Keras... ', end='', flush=True)
#from tensorflow import keras 
print('Done!')

from loss_params_qg import *
from loss_funs_qg import *
from networks_qg import *


#%% Create Model



class CustomCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        print("Starting training")
        
    def on_epoch_end(self, epoch, logs=None):
        print("Epoch training")
        
    #def on_train_batch_end(self, batch, logs=None):
        



######################## Ensemble Training ###################################
def ML_Train_Ensemble(model_type,case_name,j_ens,Nepochs,sub_sample,lstm_hist_points,dt_ml,basis_type,T):
    
    
    print('Importing Rnudged and Reference data for training... ', end='', flush=True)
    
    load_file = "../ML_Data/"+case_name+"/"+"psi1_ref.mat"
    mat_contents = sio.loadmat(load_file)
    Ny = int(mat_contents["Ny"])//1
    Nx = int(mat_contents["Nx"])//1
    data_times = int(mat_contents["data_times"])
    delta = mat_contents["delta"]
    beta  = mat_contents["beta"]
    rdrag = mat_contents["rdrag"]
    kd    = mat_contents["kd"]
    dt    = mat_contents["dt"]
    Lratio = 1.0

    print("beta =" + str(beta)+", rdrag = " + str(rdrag)+", kd =" + str(kd), end='', flush=True)
    
    psi_ref = np.zeros((Ny,Nx,data_times,2),dtype=float)
    psi_ref[:,:,:,0] = mat_contents["psi1_ref"]
    del mat_contents
    
    load_file = "../ML_Data/"+case_name+"/psi2_ref.mat"
    mat_contents = sio.loadmat(load_file)
    psi_ref[:,:,:,1] = mat_contents["psi2_ref"]
    del mat_contents
    
    
    load_file = "../ML_Data/" + case_name + "/psi1_"+ basis_type+ ".mat"
    mat_contents = sio.loadmat(load_file)
    psi_train = np.zeros((Ny,Nx,data_times,2),dtype=float)
    psi_train[:,:,:,0] = mat_contents["psi1_rnudge"]
    #psi_train[:,:,:,0] = mat_contents["psi1_" + basis_type]
    del mat_contents
    
    load_file = "../ML_Data/" + case_name + "/psi2_"+ basis_type+ ".mat"
    mat_contents = sio.loadmat(load_file)
    psi_train[:,:,:,1] = mat_contents["psi2_rnudge"]
    #psi_train[:,:,:,0] = mat_contents["psi1_" + basis_type]

    del mat_contents
    
    print('Data Import Complete!')
    
    



    # Creating Training Variables
    print('Creating training variables... ', end='', flush=True)
    print('Size of Original Data Set = '+ str(data_times), end='', flush=True)
    total_times = int(np.floor(T/dt))
    
    
    
    
    # LSTM stride to achieve desired dt for ML model
    lstm_stride = int(dt_ml/dt)
    time_start = 0
    train_times = total_times-lstm_hist_points*lstm_stride-time_start
    train_times = int(np.floor(train_times//sub_sample))
    
    print('Size of Training Data Set = '+ str(train_times), flush=True)
    print('Simulation Time Step = '+ str(dt), end='', flush=True)
    print('ML Time Step = '+ str(dt_ml), end='', flush=True)
    print('LSTM Stride = '+ str(lstm_stride), end='', flush=True)
    test_size = 1
    used_features = Ny*Nx*2
    output_dim    = 2*Ny*Nx
    input_train = np.zeros((test_size*train_times,lstm_hist_points,used_features))

    
    
    for tt in range(0,train_times):
        for jj in range(0,Ny):
            for ii in range(0,Nx):
                for pp in range(0,lstm_hist_points):
                    input_train[tt,pp,jj+Ny*ii]      = psi_train[jj,ii,sub_sample*tt+lstm_stride*pp+time_start,0]
                    input_train[tt,pp,Nx*Ny+jj+Ny*ii]= psi_train[jj,ii,sub_sample*tt+lstm_stride*pp+time_start,1]

    del psi_train
    
    
    output_train  = np.zeros((test_size*train_times,lstm_hist_points,output_dim))
    for tt in range(0,train_times):
        for jj in range(0,Ny):
            for ii in range(0,Nx):
                for pp in range(0,lstm_hist_points):
                    output_train[tt,pp,jj+Ny*ii]         = psi_ref[jj,ii,sub_sample*tt+lstm_stride*pp+time_start,0]
                    output_train[tt,pp,Ny*Nx+jj+Ny*ii]   = psi_ref[jj,ii,sub_sample*tt+lstm_stride*pp+time_start,1]
    
    del psi_ref
                
    print('Training Data Creation Done!')
    
    # Create NN Model
    os.environ['KMP_DUPLICATE_LIB_OK']='True'    
    train_end = round(test_size*train_times*0.9)
    loss_data = loss_params_2D(kd, delta, Ny, Nx, beta, rdrag)
    model1 = get_model(model_type,used_features,loss_data)
    
    Ne = 500
    Nsets = int(np.ceil(Nepochs/Ne))
    
    
    model1 = get_model(model_type,used_features,loss_data)
    for je in range(0,Nsets):
        Nepochs_j = (je+1)*Ne
        #hist1 = model1.fit(input_train[0:train_end,:,:], output_train[0:train_end,:,:], batch_size=32, epochs=Ne, validation_data=(input_train[train_end:test_size*train_times,:,:],output_train[train_end:test_size*train_times,:,:]))
        model1.fit(input_train[0:train_end,:,:], output_train[0:train_end,:,:], batch_size=32, epochs=Ne, validation_data=(input_train[train_end:test_size*train_times,:,:],output_train[train_end:test_size*train_times,:,:]))
        model1.save("../Neural_Nets/" + model_type + "_" +  basis_type+"_epochs_" + str(Nepochs_j) + "_dt_" +str(dt_ml) + "_T_" + str(T) + "_j" + str(j_ens) + ".h5")  
        
        #hist1 = model1.fit(input_train[0:train_end,:,:], output_train[0:train_end,:,:], batch_size=32, epochs=Nepochs, validation_data=(input_train[train_end:test_size*train_times,:,:],output_train[train_end:test_size*train_times,:,:]))
        #model1.save("../Neural_Nets_Topo/"+ model_type + "_" +  basis_type+"_epochs_" + str(Nepochs) + "_dt_" +str(dt_ml) + "_T_" +str(T) + ".h5")  
        #sio.savemat("../Neural_Nets_Topo/ML_Training_Hist_2D_L2_Rnudge16.mat",{"hist1":hist1.history},format='5',do_compression=True)
        #sio.savemat("../Neural_Nets_Topo/Training_History_" + model_type + "_" + basis_type+"_epochs_" + str(Nepochsj) + "_dt_" +str(dt_ml) + "_T_" +str(T) + ".mat",{"hist1":hist1.history},format='5',do_compression=True)
        
    
    
    
    
    
    del input_train, output_train
    
    return








##################### Ensemble Testing on Coarse DNS Data - Start ######################
def ML_Test_Ensemble(model_type,case_name,Nens,Nepochs,dt_ml,basis_type,compute_pdf,save_full_field,T):

    print('Importing coarse and Reference data for testing... ', end='', flush=True)
    
    load_file = "../ML_Data/"+case_name+"/"+"psi1_ref.mat"
    mat_contents = sio.loadmat(load_file)
    Ny = int(mat_contents["Ny"])//1
    Nx = int(mat_contents["Nx"])//1
    data_times = int(mat_contents["data_times"])
    delta = mat_contents["delta"]
    beta  = mat_contents["beta"]
    rdrag = mat_contents["rdrag"]
    kd    = mat_contents["kd"]
    dt    = mat_contents["dt"]
    Lratio = 1.0

    print('Data Import Complete!')
    used_features = Ny*Nx*2

    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    total_times = data_times;
    
    
 
    load_file = "../ML_Data/"+case_name+"/psi1_coarse.mat"
    mat_contents = sio.loadmat(load_file)
    psi_coarse = np.zeros((Ny,Nx,total_times,2),dtype=float)
    psi_coarse[:,:,:,0] = mat_contents["psi1_coarse"]
    data_times = int(mat_contents["data_times"])
    del mat_contents
    
    load_file = "../ML_Data/"+case_name+"/psi2_coarse.mat"
    mat_contents = sio.loadmat(load_file)
    psi_coarse[:,:,:,1] = mat_contents["psi2_coarse"]
    del mat_contents
    
    # Reshape Input Data
    input_test = np.zeros((1,total_times,used_features))   
    input_test[0,:,0:Nx*Ny]        = np.reshape(np.transpose(psi_coarse[:,:,:,0],(2,0,1)),(1,total_times,Nx*Ny), order='F')
    input_test[0,:,Nx*Ny:2*Nx*Ny]  = np.reshape(np.transpose(psi_coarse[:,:,:,1],(2,0,1)),(1,total_times,Nx*Ny), order='F')
    
    # Account for LSTM Stride
    stride = int(dt_ml/dt)
    input_test = input_test[0,0:-1:stride,:]
    input_test = np.reshape(input_test,(1,np.shape(input_test)[0],np.shape(input_test)[1]))
    test_times = np.shape(input_test)[1]
    print('Data Reshape Complete!')




    # Pdf Parameters
    Np  = int(2001)
    pleft  = -4
    pright = 4
    pdf_xaxis = np.linspace(pleft,pright,Np)
    dp = (pright-pleft)/float(Np-1)
    
    
    # preallocate pdf variables 
    psit_flag = np.zeros((Ny,Nx),dtype=float)
    psi_pdf_ML = np.zeros((Nens,Np,2),dtype=float)

    psi_rpdf_ML = np.zeros((Nens,Nx,Ny,Np,2),dtype=float)
    psi_rpdf_ref = np.zeros((Nx,Ny,Np,2),dtype=float)   
    psi_rpdf_coarse = np.zeros((Nx,Ny,Np,2),dtype=float)   
 
    # Preallocate Correlation and Spectrum Variables  
    Ni = 100
    dt_i = dt_ml*Ni
    NT = int(T/dt_ml)
    
    # Intermediate Autocorrelation Variables
    Rr = np.zeros((Ny,Nx,test_times,2),dtype=float)
    Rc = np.zeros((Ny,Nx,test_times,2),dtype=float)
    Rt = np.zeros((Ny,Nx,NT,2),dtype=float)
    Sr = np.zeros((Ny,Nx,test_times,2),dtype=float)
    Sc = np.zeros((Ny,Nx,test_times,2),dtype=float)
    St = np.zeros((Ny,Nx,NT,2),dtype=float)
   
    # Interpolated Autocorrelation and Spectra Variables
    R_ML = np.zeros((Nens,Ny,Nx,int(test_times/Ni),2),dtype=float)
    R_ref = np.zeros((Ny,Nx,int(test_times/Ni),2),dtype=float)
    R_coarse = np.zeros((Ny,Nx,int(test_times/Ni),2),dtype=float)
    S_ML = np.zeros((Nens,Ny,Nx,int(test_times/Ni),2),dtype=float)
    S_ref = np.zeros((Ny,Nx,int(test_times/Ni),2),dtype=float)
    S_coarse = np.zeros((Ny,Nx,int(test_times/Ni),2),dtype=float)   
         
    # ML Spectrum
    spectrum_ML = np.zeros((Nens,test_times,2),dtype=float)
    
    # Zonal Mean
    Psi_ML = np.zeros((Nens,Ny,test_times,2),dtype=float)
    Psi_ref = np.zeros((Ny,test_times,2),dtype=float)
    Psi_coarse = np.zeros((Ny,test_times,2),dtype=float)
    indEE_ML = np.zeros((Nens,test_times,2),dtype=float)
    indEE_ref = np.zeros((test_times,2),dtype=float)

    
    time_start = 0
    time_end   = test_times - 1
    times = time_end-time_start+1
    flag_coef = float(times*Nx*Ny)
    
    # Load Reference Data For statistics
    load_file = "../ML_Data/"+case_name+"/psi1_ref.mat"
    mat_contents = sio.loadmat(load_file)
    psi_ref = np.zeros((Ny,Nx,total_times,2),dtype=float)
    psi_ref[:,:,:,0] = mat_contents["psi1_ref"]
    del mat_contents        
    load_file = "../ML_Data/"+case_name+"/psi2_ref.mat"
    mat_contents = sio.loadmat(load_file)
    psi_ref[:,:,:,1] = mat_contents["psi2_ref"]
    del mat_contents
    load_file = "../ML_Data/"+case_name+"/psi1_coarse.mat"
    mat_contents = sio.loadmat(load_file)
    psi_coarse = np.zeros((Ny,Nx,total_times,2),dtype=float)
    psi_coarse[:,:,:,0] = mat_contents["psi1_coarse"]
    del mat_contents        
    load_file = "../ML_Data/"+case_name+"/psi2_coarse.mat"
    mat_contents = sio.loadmat(load_file)
    psi_coarse[:,:,:,1] = mat_contents["psi2_coarse"]
    del mat_contents
    # account for LSTM stride
    psi_ref = psi_ref[:,:,0:-1:stride,:]
    psi_coarse = psi_coarse[:,:,0:-1:stride,:]
    
    
    
    # compute Reference Statistics
    for l in range(0,2):
        for i in range(0,Nx):
            for j in range(0,Ny):
                # Full Version to compute spectra
                Rr[j,i,:,l] = scs.correlate(psi_ref[j,i,:,l], psi_ref[j,i,:,l], mode='same', method='auto')
                Rc[j,i,:,l] = scs.correlate(psi_coarse[j,i,:,l], psi_coarse[j,i,:,l], mode='same', method='auto')
                Rt[j,i,:,l] = scs.correlate(psi_ref[j,i,0:NT,l], psi_ref[j,i,0:NT,l], mode='same', method='auto')
                
                
                Sr[j,i,:,l] = np.abs(np.fft.fftshift(np.fft.fft(Rr[j,i,:,l])))/test_times
                Sc[j,i,:,l] = np.abs(np.fft.fftshift(np.fft.fft(Rc[j,i,:,l])))/test_times
                St[j,i,:,l] = np.abs(np.fft.fftshift(np.fft.fft(Rt[j,i,:,l])))/NT
                
                
                # interpolated version to save
                R_ref[j,i,:,l] = Rr[j,i,0:-1:Ni,l]
                R_coarse[j,i,:,l] = Rc[j,i,0:-1:Ni,l]
                S_ref[j,i,:,l] = Sr[j,i,0:-1:Ni,l]
                S_coarse[j,i,:,l] = Sc[j,i,0:-1:Ni,l]
 
                # Compute PDF
                for tt in range(time_start,time_end):
                    
                    # Reference
                    flag_valr = psi_ref[j,i,tt,l]
                    flag_pp = int((flag_valr-pleft)/dp)
                    flag_pp = max(0,flag_pp)
                    flag_pp = min(Np-1,flag_pp)
                    psi_rpdf_ref[j,i,flag_pp,l] = psi_rpdf_ref[j,i,flag_pp,l] +1
                    
                    
                    # coarse
                    flag_valc = psi_coarse[j,i,tt,l]
                    flag_pp = int((flag_valc-pleft)/dp)
                    flag_pp = max(0,flag_pp)
                    flag_pp = min(Np-1,flag_pp)
                    psi_rpdf_coarse[j,i,flag_pp,l] = psi_rpdf_coarse[j,i,flag_pp,l] +1
    # Training Data
    R_train = 1*Rt
    S_train = 1*St
    # Compute Spectra via Autocorrelation
    spectrum_ref =  np.mean(Sr,axis=(0,1))
    spectrum_coarse =  np.mean(Sc,axis=(0,1))
    spectrum_train =  np.mean(St,axis=(0,1))
    
    
    
    # Zonal Mean
    Psi_ref = np.sum(psi_ref,axis=1)/float(Nx)
    Psi_coarse = np.sum(psi_coarse,axis=1)/float(Nx)
    
    # EE Indicator
    indEE_ref = np.sum(np.abs(psi_ref),axis=(0,1))/float(Nx*Ny)
    
    # Pdf of area-over-threshold
    Npt  = int(25)
    tleft  = 0
    tright = 4
    thresh = np.linspace(tleft,tright,Npt)
    dpt = (tright-tleft)/float(Npt-1)
    
    
    aot_ref = np.zeros((Npt,test_times,2),dtype=float)
    aot_coarse = np.zeros((Npt,test_times,2),dtype=float)
    aot_ML = np.zeros((Nens,Npt,test_times,2),dtype=float)
    for l in range(0,2):
        print("max absolute value = " + str(np.max(np.abs(psi_ref[:,:,:,l]))))
        print("max value = " + str(np.max(psi_ref[:,:,:,l])))
        for tt in range(time_start,time_end):
            # loop over threshold
            for nt in range(0,Npt):
                flag_valr = np.abs(psi_ref[:,:,tt,l])
                flag_valr[flag_valr<thresh[nt]] = 0 
                flag_valr[flag_valr>=thresh[nt]] = 1 
                aot_ref[nt,tt,l] = np.sum(flag_valr)/float(Nx*Ny)
                #coarse
                flag_valc = np.abs(psi_coarse[:,:,tt,l])
                flag_valc[flag_valc<thresh[nt]] = 0 
                flag_valc[flag_valc>=thresh[nt]] = 1 
                aot_coarse[nt,tt,l] = np.sum(flag_valc)/float(Nx*Ny)
                
                
    
    
    
    
    
    del psi_ref, psi_coarse, Rr, Rc, Rt, Sr, Sc, St
    
    
    # Compute Ensemble ML Results
    for jens in range(0,Nens):
        # Evaluate Model   
        print('ML-predictions using coarse DNS data...realization # '+ str(jens), end='', flush=True) 
        model1 = tf.keras.models.load_model("../Neural_Nets/" + model_type + "_" + basis_type+"_epochs_"+str(Nepochs) + "_dt_" +str(dt_ml) + "_T_" + str(T) + "_j" + str(jens + 1) + ".h5",compile=False)
        output_test = np.zeros((1,test_times,used_features))
        output_test[:,:,:] = model1.predict(input_test)[:,:,0:used_features]

        # Reshape and Save Results
        psi_ML = np.zeros((Ny,Nx,test_times,2),dtype=float)
        Rm = np.zeros((Ny,Nx,test_times,2),dtype=float)
        Sm = np.zeros((Ny,Nx,test_times,2),dtype=float)


        psi_ML[:,:,:,0] = np.transpose(np.reshape(output_test[0,:,0:Nx*Ny],(test_times,Nx,Ny), order='F'),(1,2,0))
        psi_ML[:,:,:,1] = np.transpose(np.reshape(output_test[0,:,Nx*Ny:2*Nx*Ny],(test_times,Nx,Ny), order='F'),(1,2,0))
        del output_test
    


        # Compute Spectra via Autocorrelation
        for l in range(0,2):
            for i in range(0,Nx):
                for j in range(0,Ny):
                    # full version to compute spectra
                    Rm[j,i,:,l] = scs.correlate(psi_ML[j,i,:,l], psi_ML[j,i,:,l], mode='same', method='auto')
                    Sm[j,i,:,l] = np.abs(np.fft.fftshift(np.fft.fft(Rm[j,i,:,l])))/test_times
                    # Interpolated version to save
                    R_ML[jens,j,i,:,l] = Rm[j,i,0:-1:Ni,l]
                    S_ML[jens,j,i,:,l] = Sm[j,i,0:-1:Ni,l]
                
                    
                    # Compute PDF
                    for tt in range(time_start,time_end):
                        # ML Prediction
                        flag_valm = psi_ML[j,i,tt,l]
                        flag_pp = int((flag_valm-pleft)/dp)
                        flag_pp = max(0,flag_pp)
                        flag_pp = min(Np-1,flag_pp)
                        psi_pdf_ML[jens,flag_pp,l] = psi_pdf_ML[jens,flag_pp,l] +1
                        psi_rpdf_ML[jens,j,i,flag_pp,l] = psi_rpdf_ML[jens,j,i,flag_pp,l] +1
                        
        
        # Power Spectra
        spectrum_ML[jens,:,:] =  np.mean(Sm,axis=(0,1))
        
        # Zonal Mean
        Psi_ML[jens,:,:,:] = np.sum(psi_ML,axis=1)/float(Nx)
        
        # EE Indicator
        indEE_ML[jens,:] = np.sum(np.abs(psi_ML),axis=(0,1))/float(Nx*Ny)
        
        
        
        # Pdf of area-over-threshold
        for l in range(0,2):
            for tt in range(time_start,time_end):
                # loop over threshold
                for nt in range(0,Npt):
                    flag_valm = np.abs(psi_ML[:,:,tt,l])
                    flag_valm[flag_valm<thresh[nt]] = 0 
                    flag_valm[flag_valm>=thresh[nt]] = 1 
                    aot_ML[jens,nt,tt,l] = np.sum(flag_valm)/float(Nx*Ny)
                    
                    
                
                
        del psi_ML, Rm, Sm
   
    del input_test
    
    # Ensemble average of area-over-threshold metric
    aot_ML = np.mean(aot_ML,axis=0)
    
    # Frequency Vector
    f = np.fft.fftshift(np.fft.fftfreq(test_times,d = stride*dt))
    f_train = np.fft.fftshift(np.fft.fftfreq(NT,d = stride*dt))
    f_i = np.fft.fftshift(np.fft.fftfreq(int(test_times/Ni),d = Ni*stride*dt))
    
    
    
    
    
    
    
    # Save Stats
    save_file = "../ML_Results/"+case_name +"_"+ model_type + "_" + basis_type+"_loss_L2_epochs_"+str(Nepochs) + "_dt_" +str(dt_ml) + "_T_" +str(T) + "_Nens_" + str(Nens) + "_Ensemble_Stats_Big.mat"
    hdf5storage.savemat(save_file,{"pdf_xaxis":pdf_xaxis,'test_times':test_times,"psi_pdf_ML":psi_pdf_ML,"psi_rpdf_ML":psi_rpdf_ML,"psi_rpdf_ref":psi_rpdf_ref,"psi_rpdf_coarse":psi_rpdf_coarse,"Psi_ML":Psi_ML,"Psi_ref":Psi_ref,"Psi_coarse":Psi_coarse,"f":f,"f_train":f_train,"f_i":f_i,"spectrum_ref":spectrum_ref,"spectrum_coarse":spectrum_coarse,"spectrum_ML":spectrum_ML,"spectrum_train":spectrum_train,"R_ML":R_ML,"R_ref":R_ref,"R_coarse":R_coarse,"R_train":R_train,"S_ML":S_ML,"S_ref":S_ref,"S_coarse":S_coarse,"S_train":S_train,'dt_ml':dt_ml,'dt_i':dt_i,'thresh':thresh,'aot_ML':aot_ML,'aot_ref':aot_ref,'aot_coarse':aot_coarse,'indEE_ML':indEE_ML,'indEE_ref':indEE_ref},format='7.3',do_compression=True)
    print('Ensemble Stats Saved!')
    
    return












