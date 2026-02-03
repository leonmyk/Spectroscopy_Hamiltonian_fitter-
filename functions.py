import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path
import glob
import h5py
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys
import time
from datetime import *
from matplotlib.collections import LineCollection
import os, sys

# basic qutip imports
from qutip import *

# MCMC imports
import emcee
import corner
from tqdm import tqdm
from multiprocess import Pool

from enum import Enum
from unittest import case
import emcee
import corner
import numpy as np
from qutip import *
from multiprocess import Pool
from IPython.display import display, Math
from scipy.stats import chi2
import json

# Constants
mu_Nb = 10.4213  # [kHz / mT]
mu_Er = - 17_350 # [kHz / mT]

# Define the electron spin operators (S = 1/2)
S = 1/2
Sx = jmat(S, 'x')
Sy = jmat(S, 'y')
Sz = jmat(S, 'z')

# Define the nuclear spin operators (I = 9/2)
I = 9/2
Ix = jmat(I, 'x')
Iy = jmat(I, 'y')
Iz = jmat(I, 'z')

meas_Aperp = 51.


simu_A = np.array([[-441.66244757, -0.05970534,-6.00098845],[ -0.05970534, -441.66856909, 5.70123131],[ -6.00098845,5.70123131 ,131.30594568]])


def load_h5_to_dic(fullpath):
    with h5py.File(fullpath, 'r') as file:
        main_keys = list(file["/"].keys())
        data_vector = {}
        if isinstance(file[main_keys[0]], h5py.Dataset):
            #datasets_keys_list = [main_keys]
            for key in main_keys:
                data_vector[key]=file[key][()]
            return data_vector, main_keys
        else:
            datasets_keys_list = {}
            for j, key in enumerate(main_keys):
                datasets_keys = list(file[key].keys())
                datasets_keys_list[key]=list(file[key].keys())
                data_vector[key]={}
                for d_key in datasets_keys:
                    data_vector[key][d_key]=file[key][d_key][()]
            return data_vector, datasets_keys_list 
        
        
def complex_ramsey_fit(t,f,T,phi,A,B):
        Z=A*np.exp(1j*(2*np.pi*f*t+phi))*np.exp(-t/T) + B*(1+1j)
        return np.concatenate([np.real(Z),np.imag(Z)])
    
    
def hyperfine_hamiltonian(self,A) -> Qobj:
    h = 0 # Hyperfine interaction 
    for i, s_op in enumerate([Sx, Sy]):
        for j, i_op in enumerate([Ix, Iy, Iz]):
            h += simu_A[i, j] * tensor(s_op, i_op)
    return A * tensor(Sz, Iz) + meas_Aperp * tensor(Sz, Ix) + h


def quadrupole_hamiltonian_param(D, E, Q, delta) -> Qobj:
    q_tensor = get_q_tensor(D, E, Q, delta)
    h = 0
    for i, i1 in enumerate([Ix, Iy, Iz]):
        for j, i2 in enumerate([Ix, Iy, Iz]):
            h += q_tensor[i, j] * tensor(qeye(2), i1 * i2)
    return h

def zeeman_hamiltonian(Bz) -> Qobj:
    return -Bz * (
        mu_Er * tensor(Sz, qeye(int(2*I+1))) +
        mu_Nb * tensor(qeye(2), Iz)
    )
    

def get_q_tensor(D, E, Q, delta):
    c = E * np.cos(2 * delta)
    s = E * np.sin(2 * delta)
    q_tensor = np.array([
        [-D/2 + c,  s, Q],
        [s, -D/2 - c, 0],
        [Q, 0, D]
    ])
    return q_tensor


def get_full_q_tensor(self, D, S1, S2, delta, theta):
    cos1 = S1 * np.cos(theta)
    sin1 = S1 * np.sin(theta)
    cos2 = S2 * np.cos(2 * delta + 2 * theta)
    sin2 = S2 * np.sin(2 * delta + 2 * theta)
    q_tensor = np.array([
        [ -D/2 + cos2,        sin2, cos1],
        [        sin2, -D/2 - cos2, sin1],
        [        cos1,        sin1,    D]
    ])
    return q_tensor

def full_quadrupole_hamiltonian_param(D, S1, S2, delta, theta) -> Qobj:
    q_tensor = get_q_tensor(D, S1, S2, delta, theta)
    h = 0
    for i, i1 in enumerate([Ix, Iy, Iz]):
        for j, i2 in enumerate([Ix, Iy, Iz]):
            h += q_tensor[i, j] * tensor(qeye(2), i1*i2)
    return h


# Define the Hamiltonian
def Full_hamiltonian(x: np.ndarray) -> Qobj: 
    Bz, A, D, S1, S2, delta, alpha, Dz  = x
    return zeeman_hamiltonian(Bz) +\
        hyperfine_hamiltonian(A) +\
        full_quadrupole_hamiltonian_param(D, S1, S2, delta, alpha) +\
        sdq_hamiltonian_param(Dz) #+\
        #hexadecapole_hamiltonian(Hx)
        
        
        
def hexadecapole_hamiltonian(Hx) -> Qobj:
    # Hexadecapole term is not implemented in this context, but can be added similarly
    return Hx * tensor(Sz, Iz*Iz*Iz*Iz)

def sdq_hamiltonian_param(Dz) -> Qobj:    
    q_tensor = get_full_q_tensor(Dz, 0,0,0,0)
    h = 0
    for i, i1 in enumerate([Ix, Iy, Iz]):
        for j, i2 in enumerate([Ix, Iy, Iz]):
            h += q_tensor[i, j] * tensor(Sz, i1*i2)
    return h


def hamiltonian(x: np.ndarray) -> Qobj: 
    Bz, D, E, Q2, Q3  = x
    
    return zeeman_hamiltonian(Bz) + quadrupole_hamiltonian_param(D, E, Q2, Q3) 

def hamiltonian_Heca(x: np.ndarray) -> Qobj: 
    Bz, D, E, Q2, Q3, Hx  = x
    
    return zeeman_hamiltonian(Bz) + quadrupole_hamiltonian_param(D, E, Q2, Q3) + hexadecapole_hamiltonian(Hx)