# -*- coding: utf-8 -*-
"""
Created on Thu May  1 14:25:59 2025

@author: palchowd
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 16:49:21 2025

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
import pickle 
# from filterpy.kalman import KalmanFilter 
from scipy.signal import savgol_filter
from mpl_toolkits import mplot3d
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LogNorm


#####Location of the finalized model######
Model_Save_Directory_combined = 'C:/Users/palchowd/Datafile/Saved_Images/BiasedWeights/Combined_Materials/Logoutputs_10/'



model_pkl_file_combined = "1D_CNN_GaussianBroadenedImpulse_MAE_ShieldingThickness_150MeV_10000samples_BiasedWeights_LogValues_kernel_3_CombinedMaterials_epoch20_new.pkl"


with open(Model_Save_Directory_combined+model_pkl_file_combined, 'rb') as file:  
    model_combined = pickle.load(file) 
    



def Read_input_file_nodes(filename):
    f=open(filename, 'r')
    f1=f.readlines()
    f.close()  
    
    Neutron_Flux = []
    Neutron_Flux_Err = []
    Energy_upper = []
    Energy_lower = []
    
    for i in range(0,len(f1)):
        if f1[i] == '#  e-lower      e-upper      neutron     r.err    photon      r.err \n':
            for m in range(1,251):
                Neutron_Flux.append(float(f1[i+m].split()[2]))
                Neutron_Flux_Err.append(float(f1[i+m].split()[3]))
                Energy_lower.append(float(f1[i+m].split()[0]))
                Energy_upper.append(float(f1[i+m].split()[1])) 
    
    Neutron_Flux = np.array(Neutron_Flux)
    Neutron_Flux_Err = np.array(Neutron_Flux_Err) 
    Energy_lower = np.array(Energy_lower)
    Energy_upper = np.array(Energy_upper)
    return(Energy_lower, Energy_upper, Neutron_Flux,Neutron_Flux_Err) 

def Calculate_effective_dose_rate(Energy,Flux):
    Directory1 = 'C:/Users/palchowd/Datafile/PHITS/Inputs/Flux_Testing/'
    E, R = np.loadtxt(Directory1+"Energy_to_Effective_Dose.txt", unpack = True, delimiter= '\t') 
    
    E1 = E[31:57]
    R1 = R[31:57]

    f1 = interp1d(E1, R1)
    Response= f1(Energy)
    
    S=0 
    
    for i in range(0,len(Energy)-1):
        S= S+0.5*(Response[i])*(Flux[i+1]+Flux[i])*(Energy[i+1]-Energy[i])
    
    return(S) 

Directory1 = 'C:/Users/palchowd/Datafile/PHITS/Inputs/Flux_Testing/'
Directory2 = 'C:/Users/palchowd/Datafile/PHITS/Inputs/Flux_Testing/Flux_After_Shielding/'
Directory3 = 'C:/Users/palchowd/Datafile/PHITS/Inputs/Flux_Testing/Flux_After_Shielding_Concrete/'
# Directory4 = 'C:/Users/palchowd/Datafile/PHITS/Inputs/Flux_Testing/Ca-48_150MeV_0.1W_DoseMap/Updated/150MeV_Updated_Concrete75cm_0.1W/'


######Load the source (Ca-48 150 MeV/n beam)
E1,Weights_new2 = np.loadtxt(Directory1+"Ca-48_150MeV_Differential_Flux_NonNormalized_updated_geom_v3.dat", unpack = True, delimiter = '\t')

####Calculate the unshielded flux to compute the atteniation at different levels##########
path1 = Directory2+ 'Ca-48_150MeV_50cmVoid_v2_updated/'
filename = 'neutral_particles_shield1_HE.out'
f1,f2,f3,f4 = Read_input_file_nodes(path1+filename) 
Unshielded_flux = f3 



Concrete_thickness_sample = np.linspace(10,150,50)
Steel_thickness_sample = np.linspace(10,100,50)

G=np.zeros((50,50))

BPE_thickness = 10
Steel_thickness_sample = np.sort(Steel_thickness_sample)
Concrete_thickness_sample = np.sort(Concrete_thickness_sample)

Density1 = 1.04*np.ones(250)*0.001 ##BPE
Density2 = 2.3*np.ones(250)*0.001  ##Concrete
Density3 = 7.86*np.ones(250)*0.001 ##Steel
Weights_new2 = Weights_new2/(sum(Weights_new2))

R = np.linspace(1,250,250)

for i in range(0,len(Steel_thickness_sample)):
    for j in range(0,len(Concrete_thickness_sample)):
    
        
        Steel_thickness = Steel_thickness_sample[i]*np.ones(250)*0.001
        Concrete_thickness = Concrete_thickness_sample[j]*np.ones(250)*0.001
        BPE_thickness = BPE_thickness*np.ones(250)*0.001
        
        W = np.array([Weights_new2, Concrete_thickness,Density2]).T
        W = W.reshape(1,250,3)
        Flux = model_combined.predict(W).flatten()
     
        Flux = 10**Flux
        Steel_Flux_ML = Flux 
        a = sum(Steel_Flux_ML[20:150])/sum(Unshielded_flux[20:150])
        # a = sum(a)/len(a)
    
        # Flux[0:2] = Flux[3]
        Flux = Flux/sum(Flux)
        
        
        W = np.array([Flux, Steel_thickness,Density3]).T
        W = W.reshape(1,250,3)
        Flux1 = model_combined.predict(W).flatten()
        Flux1 = 10**Flux1 
        Concrete_Steel_Flux_ML = Flux1*a
        
    ######In case of adding the BPE layer turn this segment on#############
    
        # b = Steel_Concrete_Flux_ML[20:150]/Steel_Flux_ML[20:150]
        # b = sum(b)/len(b)
    
    
        # Flux1 = Flux1/sum(Flux1)
        
        # W = np.array([Flux1, BPE_thickness,Density1]).T
        # W = W.reshape(1,250,3)
        # Flux2 = model_combined.predict(W).flatten()
        # Flux2 = 10**Flux2  
        
        # Flux2 = Flux2*a*b
        # Effective_Dose_ML = Calculate_effective_dose_rate(R,Flux2) 
        
   ######In case of additional BPE round turn the next line off ###########
        Effective_Dose_ML = Calculate_effective_dose_rate(R,Concrete_Steel_Flux_ML)
###### This normalization factor is valid for usV/hr conversion. Change it to 8.67E7 in case of mrem/hr result
        p = Effective_Dose_ML*8.67e8
        G[i][j] = p

    



X = []
Y = []
Z = []

for  i in range(len(Steel_thickness_sample)):
    for  j in range(len(Concrete_thickness_sample)):
        #print(Steel[i], Concrete[j])
       X.append(Steel_thickness_sample[i])
       Y.append(Concrete_thickness_sample[j])
       Z.append(G[i][j])



# Z=G.T
#print(X[2499], Y[2499], Z[2499])

levels = [20,50,200,1000]

plt.figure(dpi=600)
# plt.figure(figsize=(8, 6))
plt.rc('font',family='Times New Roman')
counts,ybins,xbins,image = plt.hist2d(X, Y, bins=50, weights=Z, norm=LogNorm(),  cmap = plt.cm.rainbow) 
# plt.contour([0.1,2,5,10,100]) 
cbar = plt.colorbar()
cbar.set_label('Effective Dose Rate [$\mathregular{\mu}$Sv/h]', fontsize = 18)
cbar.ax.tick_params(labelsize=16)
cbar.set_ticks([1,1e1,1e2,1e3])
contour = plt.contour(counts.transpose(), extent=[0, 150, 0, 150], levels=levels, colors= 'black', linestyles='dashed')  
manual_locations = [(18, 20), (50, 50), (80, 130), (140,80)]
###In case of steel+concrete turn the next line on; otherwise turn it off ###
plt.clabel(contour,[20,50,200,1000], inline=True, manual= manual_locations, fontsize=18) 
# plt.clabel(contour,[20,50,200,1000], inline=True, fontsize=18) 


plt.xlabel('Steel thickness [cm] \n Upstream layer ', fontsize = 20)
plt.ylabel('Downstream layer \n Concrete thicknes [cm]', fontsize = 20)
plt.xticks([20,40,60,80,100], fontsize = 20)
plt.yticks([20,60,100,140], fontsize = 20) 
plt.ylim(10,150)
plt.xlim(10,100) 
plt.show()