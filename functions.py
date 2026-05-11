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
h    = 6.6260693e-34       # Plank constant
mu_N = 5.0507836991e-27    # Nuclear magneton in J/T
mu_0 = 12.566370614e-7     # Vacuum permeability
mu_B = 9.27401007831e-24   # Bohr magneton in J/T

# Er3+ gyromagnetic ratio
gamma_Er = np.array([117.3, 117.3, 17.45]) * 1e9 * h  # hyperfine coupling constants in Hz/T * h
g_Er = gamma_Er / mu_B
g_a, g_b, g_c = g_Er
mu_Er = - 17_350 # [kHz / mT]

# Tungsten-183 nuclear magnetic moment
gamma_W_ref = 1.77394e6 # MHz/T
mu_W = gamma_W_ref * h  # J/T
g_W = mu_W / mu_N

# Niobium-93 nuclear magnetic moment
gamma_Nb_ref = 6.567400e7/2/np.pi # MHz/T
mu_Nb_ = gamma_Nb_ref * h  # J/T
g_Nb = mu_Nb_ / mu_N

# Calcium-43 nuclear magnetic moment
gamma_Ca_ref = -2.86899e6          # Hz/T  (=-2.86899 MHz/T)  (43Ca)  :contentReference[oaicite:3]{index=3}
mu_Ca = gamma_Ca_ref * h           # J/T
g_Ca  = mu_Ca / mu_N
mu_Ca = -2.86899 #[kHz / mT]



# meas_Aperp = 51.
# meas_Aperp = 20.

# meas_Aperp = 51.
# meas_Aperp = 48.


h    = 6.6260693e-34       # Plank constant
mu_0 = 12.566370614e-7     # Vacuum permeability
gamma_Er = np.array([117.3, 117.3, 17.45]) * 1e9 * h  # hyperfine coupling constants in Hz/T * h
mu_B = 9.27401007831e-24   # Bohr magneton in J/T
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


        
def complex_ramsey_fit(t,f,T,phi,A,B):
        Z=A*np.exp(1j*(2*np.pi*f*t+phi))*np.exp(-t/T) + B*(1+1j)
        return np.concatenate([np.real(Z),np.imag(Z)])

def complex_ramsey_gaussian_fit(t,f,T,phi,A,B):
        Z=A*np.exp(1j*(2*np.pi*f*t+phi))*np.real(np.exp(-t**2/T**2)) + B*(1+1j)
        return np.concatenate([np.real(Z),np.imag(Z)])      
    

    
  
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
    if err == 0:
        return 0
    exponent = int(np.floor(np.log10(abs(err))))
    return max(0, sig_figs - 1 - exponent)


def pretty_mcmc(flat_samples, sig_figs=2, central_scale=1.0, err_scale=1.0):
    """
    Returns a list of formatted strings:
    [low_err_str, central_str, high_err_str]
    for each parameter.

    central_scale: multiply median by this before formatting
    err_scale: multiply uncertainties by this before formatting
    """
    ndim = flat_samples.shape[0]
    out = []

    for i in range(ndim):
        p16, p50, p84 = np.percentile(flat_samples[i, :], [16, 50, 84])
        q_minus, q_plus = p50 - p16, p84 - p50

        # choose decimals from the larger uncertainty after scaling
        err = max(q_minus, q_plus) * err_scale
        ndp = _sig_decimals(err, sig_figs)
        fmt = f"{{:.{ndp}f}}"

        out.append([
            fmt.format(q_minus * err_scale),
            fmt.format(p50 * central_scale),
            fmt.format(q_plus * err_scale),
        ])

    return out



