import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path
import glob
import h5py
from pathlib import Path
import numpy as np
import pandas as pd
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
mu_Ca = - 2.87

# Define the electron spin operators (S = 1/2)
S = 1/2
Sx = jmat(S, 'x')
Sy = jmat(S, 'y')
Sz = jmat(S, 'z')

# Define the nuclear spin operators (I = 9/2)
I = 7/2
Ix = jmat(I, 'x')
Iy = jmat(I, 'y')
Iz = jmat(I, 'z')


meas_Aperp = 51.
meas_Aperp = 48.


simu_A = np.array([[-441.66244757, -0.05970534,-6.00098845],[ -0.05970534, -441.66856909, 5.70123131],[ -6.00098845,5.70123131 ,131.30594568]])

def complex_ramsey_fit_n(t, *params):
    """
    Multi-frequency complex Ramsey: 
      Z(t) = sum_{i=1}^n A_i * exp[i(2π f_i t + φ_i)] * exp(-t/T) + B*(1+1j)
    params layout: [f_1..f_n, T, φ_1..φ_n, A_1..A_n, B]
    """
    n = (len(params) - 2) // 3
    freqs = params[0:n]
    T     = params[n]
    phis  = params[n+1:2*n+1]
    amps  = params[2*n+1:3*n+1]
    B     = params[-1]
    
    Z = np.zeros_like(t, dtype=complex)
    for i in range(n):
        Z += amps[i] * np.exp(1j*(2*np.pi*freqs[i]*t + phis[i])) * np.exp(-t/T)
    Z += B * (1 + 1j)
    # return concatenated real+imag for curve_fit
    return np.concatenate([Z.real, Z.imag])

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

def complex_ramsey_gaussian_fit(t,f,T,phi,A,B):
        Z=A*np.exp(1j*(2*np.pi*f*t+phi))*np.real(np.exp(-t**2/T**2)) + B*(1+1j)
        return np.concatenate([np.real(Z),np.imag(Z)])      
    
def hyperfine_hamiltonian(A) -> Qobj:
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
        mu_Ca * tensor(qeye(2), Iz)
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


def get_full_q_tensor(D, S1, S2, delta, theta):
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
    q_tensor = get_full_q_tensor(D, S1, S2, delta, theta)
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


    
def normalise_Histogram_Height(data1,data2,bins1,bins2):

    # choose bins independently (examples)
    edges1 = np.histogram_bin_edges(data1, bins=bins1)     # or 'fd', 'auto', etc.
    edges2 = np.histogram_bin_edges(data2, bins=bins2)

    c1, e1 = np.histogram(data1, bins=edges1)
    c2, e2 = np.histogram(data2, bins=edges2)

    c1 = c1 / c1.max()
    c2 = c2 / c2.max()

    w1 = np.diff(e1)
    w2 = np.diff(e2)
    
    return (e1[:-1], c1, w1),(e2[:-1], c2, w2)

def _sig_decimals(err, sig_figs=2):
    """
    Return the number of decimals needed to keep `sig_figs`
    significant figures of an uncertainty `err`.

    Example:
    err = 0.0123  -> 3  (0.012 shown with 2 s.f.)
    err = 3.4     -> -1 ( 3  shown with 2 s.f.)
    """
    if err == 0:
        return 0
    exponent = int(np.floor(np.log10(err)))
    return max(0, sig_figs - 1 - exponent)

def pretty_mcmc(flat_samples, sig_figs=2):
    """
    Print median and asymmetric 1-sigma errors with only the
    relevant digits for each parameter.
    """
    ndim = flat_samples.shape[1]
    high_low = np.zeros((ndim, 3))

    for i in range(ndim):
        p16, p50, p84 = np.percentile(flat_samples[:, i], [16, 50, 84])
        q_minus, q_plus = p50 - p16, p84 - p50
        # Use the larger side as a conservative uncertainty
        err = max(q_minus, q_plus)
        ndp = _sig_decimals(err, sig_figs)

        fmt = f"{{:.{ndp}f}}"
        central = fmt.format(p50)
        low    = fmt.format(q_minus)
        high   = fmt.format(q_plus)
        high_low[i] = [low, central,high]

    return high_low

    
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