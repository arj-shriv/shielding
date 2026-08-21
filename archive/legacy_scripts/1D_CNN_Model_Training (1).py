# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 09:23:53 2024

@author: palchowd
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 22 13:05:56 2024

@author: palchowd
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 17:03:18 2024

@author: palchowd
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score 
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import scipy
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from scipy import interpolate
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow.keras as tf 
from keras.layers import Dense, Reshape, Conv1D, Conv2D, MaxPooling2D, Conv1DTranspose, Flatten, MaxPooling1D,Dropout, Activation, BatchNormalization,Input, AveragePooling1D
from sklearn.svm import SVR
import pickle
# from d2l import tensorflow as d2l


# This is the location where the trained model will be saved ##
Model_Save_Directory  = 'C:/Users/palchowd/Datafile/Saved_Images/BiasedWeights/Combined_Materials/Logoutputs_10/'


def Response_Creation(Material_Directory, Thickness_List):
    
    
    Y1 = []
    for i in range(0, len(Thickness_List)):
        Flux_Output = np.loadtxt(Material_Directory +
                                 str(int(Thickness_List[i]))+'cm/'+"Neutron_Diff_Flux_Response.dat")
        
        Flux_Output= Flux_Output.T
        for j in range(0,250):
            
            A = list(Flux_Output[j]) 
            Y1.append(A)
    
    Y1 = np.array(Y1)
    Y1 = Y1.reshape(len(Thickness_List),250,250)
    # Y1 = Y1.reshape(250)
    
    Y2 = []
    for i in range(0,len(Thickness_List)):
        Y2.append(Y1[i].T)
    
    Y2 = np.array(Y2).T
    
    Y2 = Y2/(200*200*50)     
    return Y2

def generate_output_flux(Y, Thickness_List, Density):
 


    Y_new = []
    X_new = []    
    

    #Creating the database for training the system by linearly combining the output flux#
    for k in range(0,len(Thickness_List)):
        for i in range(0,10000):
            
            
            ##############For updated weight mechanism ##############
            S = np.zeros(int(Y.shape[1])) 
            E = np.linspace(0,1,250)
            B = np.ones(250)
            B1 = np.ones(250)
            Position_index = []
            A = np.zeros(int(Y.shape[1])) 
            w1 = np.random.randint(50,200) 
            Weights = np.random.rand(w1)
            # Weights = Weights/sum(Weights) 
            

            for j in range(0,w1):
                Position_index.append(np.random.randint(0,249))
            for j in range(0,w1):
                A[int(Position_index[j])] = Weights[j]
           
            # For 150 MeV/n Ca-48
            Weights_New = 0.003/(1+np.exp((E-0.6)/0.08))
            # For 200 MeV/n Ca-48
            # Weights_New =  0.003/(1+np.exp((E-0.65)/0.13))
            # Weights_New = 0.003/(1+np.exp((E-0.67)/0.14))
            #For 250 MeV/n Ca-48 
            # Weights_New = 0.005/(1+np.exp((E-0.96)/0.17))
            
            for j in range(0,250):
                if A[j] !=0:
                    A[j] = Weights_New[j]
                else:
                    A[j] = 1e-7
            
            A = A/sum(A) 
            B = B*Thickness_List[k]*0.001
            B1 = B1*Density*0.001
            C = np.array([A,B,B1]).T    
            
            for j in range(0,len(A)):
                S = S + Y[j,:,k]*A[j]
            
            S1 = np.zeros(250)
            for j in range(0,250):
                if S[j] != 0: 
                    S1[j] = np.log10(S[j])
                
            
            Y_new.append(S1)
            X_new.append(C)
    
    
    
    X_new = np.array(X_new)
    Y_new = np.array(Y_new)


    return(X_new, Y_new)

#########Setting up the ML Algorithm##########

New_Thickness = np.array([5])
Thickness_List = np.linspace(10, 150,15)
Thickness_List1 = np.array(list(New_Thickness)+list(Thickness_List))


Thickness_List2 = np.linspace(10,100,10)
# Thickness_List = np.array([10,20,30,50,60,80,100,150])
Thickness_Results_Directory1 = 'C:/Users/palchowd/Datafile/PHITS/Inputs/Neutron_Flux_Shielding_Thickness/Concrete/'
Thickness_Results_Directory2 = 'C:/Users/palchowd/Datafile/PHITS/Inputs/Neutron_Flux_Shielding_Thickness/Steel/'
Thickness_Results_Directory3 = 'C:/Users/palchowd/Datafile/PHITS/Inputs/Neutron_Flux_Shielding_Thickness/BPE/'


#############Define Densities###########

Density_concrete = 2.3
Density_steel = 7.86
Density_BPE = 1.04

Y_Concrete = Response_Creation(Thickness_Results_Directory1, Thickness_List1)
Y_Steel = Response_Creation(Thickness_Results_Directory2, Thickness_List2)
Y_BPE = Response_Creation(Thickness_Results_Directory3, Thickness_List2)


x_concrete,y_concrete =  generate_output_flux(Y_Concrete,Thickness_List1, Density_concrete)

## This section creates the input and output for the training containing 16 thicknesses of concrete (5 to 150 cm) each with 10000 i/p and corresponding o/p spectra ##  
## x contains the input containing input strengths, thickness, and density; y contains an array representing the output flux
#Each input and output spectra consists of an array with length equal to 250 representing the number of energy bins simulated with PHITS
x_concrete =x_concrete.reshape(160000,250,3)
y_concrete= y_concrete.reshape(160000,250,1)   

## This section creates the input and output for the training containing 10 thicknesses of steel (10 to 100 cm) each with 10000 i/p and o/p spectra ## 
## x contains the input containing input strengths, thickness, and density; y contains an array representing the output flux
#Each input and output spectra consists of an array with length equal to 250 representing the number of energy bins simulated with PHITS 
x_steel,y_steel =  generate_output_flux(Y_Steel,Thickness_List2, Density_steel)
   
x_steel =x_steel.reshape(100000,250,3)
y_steel = y_steel.reshape(100000,250,1)

## This section creates the input and output for the training containing 10 thicknesses of BPE (10 to 100 cm) each with 10000 i/p and o/p spectra ## 
## x contains the input containing input strengths, thickness, and density; y contains an array representing the output flux
#Each input and output spectra consists of an array with length equal to 250 representing the number of energy bins simulated with PHITS 
x_BPE,y_BPE =  generate_output_flux(Y_BPE,Thickness_List2, Density_BPE)
   
x_BPE = x_BPE.reshape(100000,250,3)
y_BPE = y_BPE.reshape(100000,250,1) 

##This part combines all the responses for three materials for energies ranging from 1-250 MeV together
x = np.array(list(x_concrete)+list(x_steel)+list(x_BPE))
y = np.array(list(y_concrete)+list(y_steel)+list(y_BPE))



xtrain, xtest, ytrain, ytest=train_test_split(x, y, test_size=0.3)


def build_conv1d_model():
    sample_length = xtrain.shape[1] 
    sample_features = xtrain.shape[2] 
    model = Sequential() 
    model.add(Input(shape = (sample_length, sample_features)))
    model.add(Conv1D(filters= 32, kernel_size= 3, activation = 'relu',padding = 'same', name = 'Conv1D_1'))
    model.add(Conv1D(filters= 64, kernel_size= 3, activation = 'relu',padding = 'same', name = 'Conv1D_2'))
    model.add(MaxPooling1D(pool_size =2, padding = 'same'))
    model.add(Flatten())
    model.add(Dense(512, name = 'Dense_1'))
    model.add(Dense(sample_length))
    model.compile(optimizer='adam',loss='mae',metrics=['mean_absolute_error'])
    return model 





model_conv1D = build_conv1d_model()
    
model_conv1D.summary()

history = model_conv1D.fit(xtrain, ytrain, epochs=20, validation_split= 0.3, verbose=2) 

####Save the trained model#######
model_pkl_file = "1D_CNN_GaussianBroadenedImpulse_MAE_ShieldingThickness_150MeV_10000samples_BiasedWeights_LogValues_kernel_3_CombinedMaterials_epoch20_new.pkl"
with open(Model_Save_Directory + model_pkl_file, 'wb') as file:  
    pickle.dump(model_conv1D, file)




####This segment plots the loss function######
plt.figure(1)
plt.figure(dpi=600)
plt.rc('font',family='Times New Roman')
plt.plot(history.history['loss'], label = 'Training_loss', linestyle = 'solid')
plt.plot(history.history['val_loss'], label = 'Validation_loss', linestyle = 'dashed')
# plt.title('model loss')
plt.ylabel('Loss (arb.u)', fontsize = 16)
plt.xlabel('Number of Epochs', fontsize = 16) 
plt.legend(['Training', 'Validation'], loc='upper right', fontsize = 16)
plt.xticks([0,5,10,15,20], fontsize = 16) 
plt.yticks(fontsize= 16)
plt.show()


R = np.linspace(1,250,250)

y_predict = model_conv1D.predict(xtest).flatten() 


#####This section plots some example comparison between test and train section####

for i in range(0,20):
    pred_y = y_predict[250*i:250*(i+1)]
    plt.figure(i+1)
    plt.figure(dpi=600)
    plt.rc('font',family='Times New Roman')
    plt.plot(R, pred_y,color = 'red', label = 'ML_predicted', linestyle = 'dashed')
    plt.plot(R, ytest[i], color = 'black', label = 'Test_output', linestyle = 'solid')
    plt.xlabel("Energy (MeV)", fontsize = 16)
    plt.ylabel("Neutron Differential Flux (arb.u) ", fontsize = 16)
    plt.legend(fontsize = 16, loc = 'best')  
    plt.xticks (fontsize = 16)
    plt.yticks(fontsize = 16)      
    # plt.yscale('log')  
    plt.savefig(Model_Save_Directory+'image'+str(i+1))


##### This segment plots an arbitrary spectra ####
plt.figure(1)
plt.figure(dpi=600)
plt.rc('font',family='Times New Roman')
plt.semilogy(R, xtrain[15000,:,0])
plt.ylabel('Normalized Input Strengths', fontsize = 16)
plt.xlabel('Energy (MeV)', fontsize = 16) 
plt.show()

