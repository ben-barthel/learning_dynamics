This repository includes the code needed to train and test the machine learning models described in Barthel Sorensen et.al. 2024. The code is implemented in Python using TensorFlow. 


DATA
==================================================================================
The training and testing trajectories can be found at: www.data.com. The training data is contained in the folder "QG_beta2.0_rdrag0.1_realization1_24x24", it contains the reference, coarse, and spectrally corrected nudged trajectories of length 10,000 time units. The test data is contained in "QG_beta2.0_rdrag0.1_realization2_24x24" it contains the reference and coarse data of length 34,000 time units. The following codes assume that these two folders are located in the folder "ML_Data".


TRAINING
==================================================================================
To train one iteration of the ML model, run the script ML_Code/python master_train_qg.py.

The script master_train_qg.py takes 4 user inputs. m T N j

m: model architecture: "LSTM" "VAE-RNN", "STORN", or "VRNN" 

T: length of training data: T_{train} eg. 1,000 to reproduce results from paper (max 10,000 available)

N: number of epochs

j: Ensemble member index

The trained networks are saved in the folder "Neural_Nets"

For example, to train the LSTM on 1000 time units of data for 500 epochs with ensemble index 3 run: python master_train_qg.py 'LSTM' 1000 500 3


TESTING
==================================================================================
To test an ensemble of ML models, run the script ML_Code/python master_test_qg.py.

The script master_test_qg.py takes 4 user inputs. m T N Nj

m: model architecture: "LSTM" "VAE-RNN", "STORN", or "VRNN"

T: length of training data: T_{train} eg. 1,000 to reproduce results from paper

N: number of epochs

Nj: Ensemble size. (you must train models with index j = 1-Nj first)

The ML predictions are saved in the folder "ML_Results"

For example to test an ensemble of 6 VRNN models trained on 1000 time units of data for 500 epochs run: python master_test_qg.py 'VRNN' 1000 500 6
