# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 13:39:12 2025

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
import pylab
import matplotlib.ticker as tck
from matplotlib.ticker import MultipleLocator, LogLocator, NullLocator, AutoMinorLocator, NullFormatter



###Location of the stored finalized CNN model#####

Model_Save_Directory_combined = 'C:/Users/palchowd/Datafile/Saved_Images/BiasedWeights/Combined_Materials/Logoutputs_10/'

#Present location of the stored model#
#Model_Save_Directory_combined = '\\thumper7.nscl.msu.edu\evtdata\radtrans\Raj\Important_Immediate_Files\Saved_Images\BiasedWeights\Combined_Materials\Logoutputs_10\'

Directory1 = 'C:/Users/palchowd/Datafile/PHITS/Inputs/Flux_Testing/'

# This is where the actual files are located at present # 
#Directory1 = '\\thumper7.nscl.msu.edu\evtdata\radtrans\Raj\Important_Immediate_Files\Flux_Testing'

model_pkl_file_combined = "1D_CNN_GaussianBroadenedImpulse_MAE_ShieldingThickness_150MeV_10000samples_BiasedWeights_LogValues_kernel_3_CombinedMaterials_epoch20_new.pkl"


with open(Model_Save_Directory_combined+model_pkl_file_combined, 'rb') as file:  
    model_combined = pickle.load(file)


#### This function reads the flux information generated using PHITS#####
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



#All the files in Directory1, Directory2, and Directory3 are present here now:
#Directory1 = '\\thumper7.nscl.msu.edu\evtdata\radtrans\Raj\Important_Immediate_Files\Flux_Testing\'

Directory1 = 'C:/Users/palchowd/Datafile/PHITS/Inputs/Flux_Testing/'
Directory2 = 'C:/Users/palchowd/Datafile/PHITS/Inputs/Flux_Testing/Flux_After_Shielding/'
Directory3 = 'C:/Users/palchowd/Datafile/PHITS/Inputs/Flux_Testing/Flux_After_Shielding_Concrete/'


###Loads the source spctra for Ca-48 150 MeV/n beam harvested at a 1 cm^3 volume#### 
E1,Weights_new1 = np.loadtxt(Directory1+"Ca-48_150MeV_Differential_Flux_NonNormalized_updated_geom_v3.dat", unpack = True, delimiter = '\t')


##### Computes the unshielded flux for the purpose of computing the attenuation of it at different levels#########
path7 = Directory3+'Ca-48_150MeV_'+str(50) +'cmVoid_v2_updated/' 

filename = 'neutral_particles_shield1_HE.out'

f1,f2,f3,f4 = Read_input_file_nodes(path7+filename) 

Unshielded_flux = f3

########Computest the thickness and density vectors subjected to different shielding materials for passing to the CNN model###########

###These are the thicknesses for a fixed material concrete: For generating Figure 4 in the paper###

Thickness1 = 25*np.ones(250)*0.001
Thickness2 = 75*np.ones(250)*0.001
Thickness3 = 112*np.ones(250)*0.001 
Thickness4 = 145*np.ones(250)*0.001 


###This thicknessis fixed for every material: For generating Figure 5 in the paper####
Thickness = 65*np.ones(250)*0.001


### These are the density vectors used for the CNN model for generating Figure 4 and 5#### 
Density1 = 1.04*np.ones(250)*0.001  ### BPE
Density2 = 2.3*np.ones(250)*0.001   ##Concrete
Density3 = 7.86*np.ones(250)*0.001  ## Steel
Density4 = 4.0*np.ones(250)*0.001   ## HD-Concrete


### Normalize the source so that the sum of the input strengths is equal to 1###
Weights_new1 = Weights_new1/sum(Weights_new1)




### Generate CNN prediction for different materials but for a fixed thickness ###
W = np.array([Weights_new1, Thickness, Density4]).T
W = W.reshape(1,250,3)
Flux = model_combined.predict(W).flatten()
Flux = 10**Flux
HDConcrete_65cm_ML = Flux  


W = np.array([Weights_new1, Thickness, Density1]).T
W = W.reshape(1,250,3)
Flux = model_combined.predict(W).flatten()
Flux = 10**Flux
BPE_65cm_ML = Flux   

W = np.array([Weights_new1, Thickness, Density3]).T
W = W.reshape(1,250,3)
Flux = model_combined.predict(W).flatten()
Flux = 10**Flux
Steel_65cm_ML = Flux  

W = np.array([Weights_new1, Thickness, Density2]).T
W = W.reshape(1,250,3)
Flux = model_combined.predict(W).flatten()
Flux = 10**Flux
Concrete_65cm_ML = Flux  

### Generate CNN prediction for different thicknesses but for a fixed material ###

W = np.array([Weights_new1, Thickness1, Density2]).T
W = W.reshape(1,250,3)
Flux = model_combined.predict(W).flatten()
Flux = 10**Flux
Concrete_25cm_ML = Flux

W = np.array([Weights_new1, Thickness2, Density2]).T
W = W.reshape(1,250,3)
Flux = model_combined.predict(W).flatten()
Flux = 10**Flux
Concrete_75cm_ML = Flux  

W = np.array([Weights_new1, Thickness3, Density2]).T
W = W.reshape(1,250,3)
Flux = model_combined.predict(W).flatten()
Flux = 10**Flux
Concrete_112cm_ML = Flux  

W = np.array([Weights_new1, Thickness4, Density2]).T
W = W.reshape(1,250,3)
Flux = model_combined.predict(W).flatten()
Flux = 10**Flux
Concrete_145cm_ML = Flux 







###########This section loads the output flux for different material and thickness combinations generated by PHITS########

path1 = Directory2+'Ca-48_150MeV_65cmBPE_v2/'

filename = 'neutral_particles_shield1_HE.out'
   

f1,f2,f3,f4 = Read_input_file_nodes(path1+filename) 

BPE_65cm_PHITS = f3 
BPE_65cm_PHITS_Err = f4

path1 = Directory3+'Ca-48_150MeV_25cmConcrete_v2_updated/'

filename = 'neutral_particles_shield1_HE.out'
   

f1,f2,f3,f4 = Read_input_file_nodes(path1+filename) 

Concrete_25cm_PHITS = f3 
Concrete_25cm_PHITS_Err = f4 



path1 = Directory3+'Ca-48_150MeV_75cmConcrete_v2_updated/'

filename = 'neutral_particles_shield1_HE.out'
   

f1,f2,f3,f4 = Read_input_file_nodes(path1+filename) 

Concrete_75cm_PHITS = f3
Concrete_75cm_PHITS_Err = f4


path1 = Directory2+'Ca-48_150MeV_65cmHDConcrete_v2_updated/'

filename = 'neutral_particles_shield1_HE.out'
   

f1,f2,f3,f4 = Read_input_file_nodes(path1+filename) 

HDConcrete_65cm_PHITS = f3 
HDConcrete_65cm_PHITS_Err = f4  

path1 = Directory2+'Ca-48_150MeV_37cmHDConcrete_v2_updated/'

filename = 'neutral_particles_shield1_HE.out'
   

f1,f2,f3,f4 = Read_input_file_nodes(path1+filename) 

HDConcrete_37cm_PHITS = f3 
HDConcrete_37cm_PHITS_Err = f4 


path1 = Directory2+'Ca-48_150MeV_75cmHDConcrete_v2_updated/'

filename = 'neutral_particles_shield1_HE.out'
   

f1,f2,f3,f4 = Read_input_file_nodes(path1+filename) 

HDConcrete_75cm_PHITS = f3 
HDConcrete_75cm_PHITS_Err = f4 


path1 = Directory2+'Ca-48_150MeV_105cmHDConcrete_v2_updated/'

filename = 'neutral_particles_shield1_HE.out'
   

f1,f2,f3,f4 = Read_input_file_nodes(path1+filename) 

HDConcrete_105cm_PHITS = f3 
HDConcrete_105cm_PHITS_Err = f4 

path1 = Directory2+'Ca-48_150MeV_145cmHDConcrete_v2_updated/'

filename = 'neutral_particles_shield1_HE.out'
   

f1,f2,f3,f4 = Read_input_file_nodes(path1+filename) 

HDConcrete_145cm_PHITS = f3 
HDConcrete_145cm_PHITS_Err = f4 

path1 = Directory2+'Ca-48_150MeV_65cmHDBPE_v2_updated/'

filename = 'neutral_particles_shield1_HE.out'
   

f1,f2,f3,f4 = Read_input_file_nodes(path1+filename) 

HDBPE_65cm_PHITS = f3 
HDBPE_65cm_PHITS_Err = f4 


path1 = Directory2+'Ca-48_150MeV_65cmAluminum_v2_updated/'

filename = 'neutral_particles_shield1_HE.out'
   

f1,f2,f3,f4 = Read_input_file_nodes(path1+filename) 

Aluminum_65cm_PHITS = f3 
Aluminum_65cm_PHITS_Err = f4 

path1 = Directory2+'Ca-48_150MeV_65cmCastIron_v2_updated/'

filename = 'neutral_particles_shield1_HE.out'
   

f1,f2,f3,f4 = Read_input_file_nodes(path1+filename) 

CastIron_65cm_PHITS = f3 
CastIron_65cm_PHITS_Err = f4 


path1 = Directory3+'Ca-48_150MeV_75cmConcrete_v2_updated/'

filename = 'neutral_particles_shield1_HE.out'
   

f1,f2,f3,f4 = Read_input_file_nodes(path1+filename) 

Concrete_75cmFlux = f3

path1 = Directory3+'Ca-48_150MeV_100cmConcrete_v2_updated/'

filename = 'neutral_particles_shield1_HE.out'
   

f1,f2,f3,f4 = Read_input_file_nodes(path1+filename) 

Concrete_100cmFlux = f3



path1 = Directory2+'Ca-48_150MeV_50cmConcrete_65cmSteel_v2_updated/'

filename = 'neutral_particles_shield1_HE.out'
   

f1,f2,f3,f4 = Read_input_file_nodes(path1+filename) 

Concrete_50cm_Steel_65cmFlux = f3

path2 = Directory2+'Ca-48_150MeV_100cmConcrete_45cmSteel_v2_updated/'

   

f1,f2,f3,f4 = Read_input_file_nodes(path2+filename) 

Concrete_100cm_Steel_45cmFlux = f3 

path3 = Directory2+'Ca-48_150MeV_75cmSteel_50cmConcrete_v2_updated/'


f1,f2,f3,f4 = Read_input_file_nodes(path3+filename) 

Steel_75cm_Concrete_50cmFlux = f3 

   
path4 = Directory2+'Ca-48_150MeV_55cmSteel_60cmConcrete_v2_updated/'

f1,f2,f3,f4 = Read_input_file_nodes(path4+filename) 

Steel_55cm_Concrete_60cmFlux = f3 


path5 = Directory2+'Ca-48_150MeV_75cmConcrete_80cmBPE_v2_updated/'

f1,f2,f3,f4 = Read_input_file_nodes(path5+filename) 

Concrete_75cm_BPE_80cmFlux = f3 

path6 = Directory2+'Ca-48_150MeV_60cmSteel_50cmBPE_v2_updated/'

f1,f2,f3,f4 = Read_input_file_nodes(path6+filename) 

Steel_60cm_BPE_50cmFlux = f3 




path8 = Directory3+'test_function_Ca-48_150MeV_145cm/'

filename = 'neutral_particles_shield1_HE.out'
   

f1,f2,f3,f4 = Read_input_file_nodes(path8+filename) 

Concrete_145cm_PHITS = f3 
Concrete_145cm_PHITS_Err = f4 


path8 = Directory3+'test_function_Ca-48_150MeV_112cm/'

filename = 'neutral_particles_shield1_HE.out'
   

f1,f2,f3,f4 = Read_input_file_nodes(path8+filename) 

Concrete_112cm_PHITS = f3 
Concrete_112cm_PHITS_Err = f4 



path8 = Directory2+'Ca-48_150MeV_65cmSteel_v2_updated/'

filename = 'neutral_particles_shield1_HE.out'
   

f1,f2,f3,f4 = Read_input_file_nodes(path8+filename) 

Steel_65cm_PHITS = f3
Steel_65cm_PHITS_Err = f4



path8 = Directory2+'Ca-48_150MeV_65cmConcrete_v2_updated/'

filename = 'neutral_particles_shield1_HE.out'
   

f1,f2,f3,f4 = Read_input_file_nodes(path8+filename) 

Concrete_65cm_PHITS = f3
Concrete_65cm_PHITS_Err = f4




#########Generate the plot for Figure 4, turn on or off ######
# plt.rcParams['ytick.minor.visible'] = True
# fig, ax = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(8, 6))
# fig.supxlabel('Energy [MeV]', fontsize = 22, fontname = 'Times New Roman')
# fig.supylabel("      Differential Flux [$\mathregular{{cm}^{-2} \\ {MeV}^{-1} \\ {source}^{-1}}$]", fontsize = 22, fontname = 'Times New Roman')


# R = np.linspace(1,250,250)

# ax[0, 0].semilogy(R,Concrete_25cm_ML, color = 'r', linestyle = 'dashed' )
# ax[0, 0].semilogy(R,Concrete_25cm_PHITS, color = 'black', linestyle = 'solid') 

# ax[0, 1].semilogy(R,Concrete_75cm_ML, color = 'r', linestyle = 'dashed' )
# ax[0, 1].semilogy(R,Concrete_75cm_PHITS, color = 'black', linestyle = 'solid') 

# ax[1, 0].semilogy(R,Concrete_112cm_ML, color = 'r', linestyle = 'dashed')
# ax[1, 0].semilogy(R,Concrete_112cm_PHITS/(50*200*200), color = 'black', linestyle = 'solid') 

# ax[1, 1].semilogy(R,Concrete_145cm_ML, color = 'r', linestyle = 'dashed')
# ax[1, 1].semilogy(R,Concrete_145cm_PHITS/(50*200*200), color = 'black', linestyle = 'solid')



# ax[0, 0].tick_params(which = 'both', axis='both', labelsize=20)
# ax[0, 1].tick_params(axis='both', labelsize=20)
# ax[1, 0].tick_params(which = 'both', axis='both', labelsize=20)
# ax[1, 1].tick_params(axis='both', labelsize=20)


# ax[0,0].text(20, 0.8e-5, '25 cm', fontsize=24, fontname = 'Times New Roman')
# ax[0,1].text(20, 0.8e-5, '75 cm', fontsize=24, fontname = 'Times New Roman')
# ax[1,0].text(20, 0.8e-5, '112 cm', fontsize=24, fontname = 'Times New Roman')
# ax[1,1].text(20, 0.8e-5, '145 cm', fontsize=24, fontname = 'Times New Roman')



# plt.setp(ax, xticks=[0,50,100,150,200,250], yticks=[1e-10,1e-8,1e-6,1e-4])
# fig.legend(['CNN','PHITS'], loc='lower center', bbox_to_anchor=(0.55, 0.91), ncol=2, fontsize = 22, frameon=False)
# fig.tight_layout()
# fig.set_dpi(800)



#########Generate the plot for Figure 5, turn on or off######

plt.rcParams['ytick.minor.visible'] = True
fig, ax = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(8, 6))
fig.supxlabel('Energy [MeV]', fontsize = 22, fontname = 'Times New Roman')
fig.supylabel("     Differential Flux [$\mathregular{{cm}^{-2} \\ {MeV}^{-1} \\ {source}^{-1}}$]", fontsize = 22, fontname = 'Times New Roman')


R = np.linspace(1,250,250)

ax[0, 0].semilogy(R,BPE_65cm_ML, color = 'r', linestyle = 'dashed' )
ax[0, 0].semilogy(R,BPE_65cm_PHITS, color = 'black', linestyle = 'solid') 

ax[0, 1].semilogy(R,Concrete_65cm_ML, color = 'r', linestyle = 'dashed' )
ax[0, 1].semilogy(R,Concrete_65cm_PHITS, color = 'black', linestyle = 'solid' ) 

ax[1, 0].semilogy(R,Steel_65cm_ML, color = 'r', linestyle = 'dashed')
ax[1, 0].semilogy(R,Steel_65cm_PHITS, color = 'black', linestyle = 'solid') 

ax[1, 1].semilogy(R,HDConcrete_65cm_ML, color = 'r', linestyle = 'dashed')
ax[1, 1].semilogy(R,HDConcrete_65cm_PHITS, color = 'black', linestyle = 'solid') 

ax[0,0].text(20, 0.8e-5, 'BPE', fontsize=24, fontname = 'Times New Roman')
ax[0,1].text(20, 0.8e-5, 'Concrete', fontsize=24, fontname = 'Times New Roman')
ax[1,0].text(20, 0.8e-5, 'Steel', fontsize=24, fontname = 'Times New Roman')
ax[1,1].text(20, 0.8e-5, 'HD Concrete', fontsize=24, fontname = 'Times New Roman')

ax[0, 0].tick_params(which = 'both', axis='both', labelsize=20)
ax[0, 1].tick_params(axis='both', labelsize=20)
ax[1, 0].tick_params(which = 'both', axis='both', labelsize=20)
ax[1, 1].tick_params(axis='both', labelsize=20)


plt.setp(ax, xticks=[0,50,100,150,200,250], yticks=[1e-10,1e-8,1e-6,1e-4])
fig.legend(['CNN','PHITS'], loc='lower center', bbox_to_anchor=(0.55, 0.91), ncol=2, fontsize = 22, frameon=False)
fig.tight_layout()
fig.set_dpi(800)

