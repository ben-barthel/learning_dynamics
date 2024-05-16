# learning_dynamics
This repository includes the code needed to train and test the machine learning models described in Barthel Sorensen et.al. 2024. The code is implemented in Python using TensorFlow. 
The data can be found at: www.data.com, and the following codes assume the data is located in the folder "ML_Data"


==================================================================================
TRAINING
==================================================================================
To train one iteration of the ML model, run the script ML_Code/python master_train_qg.py
master_train_qg.py takes 4 user inputs. m T N j


m: model architecture: "LSTM" "VAE-RNN", "STORN", or "VRNN"
T: length of training data: T_{train} eg. 1,000 to reproduce results from paper (max 10,000 available)
N: number of epochs
j: Ensemble member index

The trained networks are saved in the folder "Neural_Nets"

For example, to train the LSTM NN on 1000 time units of data for 500 epochs with ensemble index 3 run:python master_train_qg.py 'LSTM' 1000 500 3


==================================================================================
TESTING
==================================================================================
To test an ensemble of ML models, run the script ML_Code/python master_test_qg.py
master_test_qg.py takes 4 user inputs. m T N Nj


m: model architecture: "LSTM" "VAE-RNN", "STORN", or "VRNN"
T: length of training data: T_{train} eg. 1,000 to reproduce results from paper
N: number of epochs
Nj: Ensemble size. (you must train models with index j = 1-Nj first)

The ML predictions are saved in the folder "ML_Results"

For example to test an ensemble of 6 VRNN models trained on 1000 time units of data for 500 epochs run: python master_test_qg.py 'VRNN' 1000 500 7
