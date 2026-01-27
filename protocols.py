# basic qutip imports
import numpy as np
from qutip import *
import matplotlib.pyplot as plt

# MCMC imports
import emcee
import corner
from tqdm import tqdm
from multiprocess import Pool

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


ground_meas_Nb = np.array([7560562 ,6894745 ,6227459 , 5557760 , 4883861.9 ,4202128.8 , 3504133.1, 2774604.8, 1822491.2]) * 1e-3
manu_ramsey_meas_Nb = np.array([-136547.1, -135922.4, -135196.4, -134203.1, -132831.5, -130986.4, -128470.1, -122551.2, -128756.7])* 1e-3

# ground_meas = np.array([ 905.88797078,  996.60227871, 1089.83895067, 1184.69640193,
#        1280.6226584 , 1377.27235175, 1474.42393934, 1571.9317065 ,
#        1669.69749955])
# ground_meas = np.array([9.97843441e+02, 1.09079304e+03, 1.18544236e+03, 1.28121530e+03,1.37775014e+03, 1.47481429e+03, 1.57225446e+03, 7.79851989e+06,9.97843441e+02])
# ground_meas = np.array([997843.4 ,1090793.4 ,1185442.1 ,1281215.1 ,1377750.3 ,1474814.6 ,1572254.3]) * 1e-3

# manu_ramsey_meas = np.array([-27703.3,-27670.2,-27594.3,-27518.5,-27449.3,-27376.6,-27349.6]) * 1e-3
# excited_meas = manu_ramsey_meas + ground_meas

d_ground_meas = np.array([0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]) * 1e-3 # [kHz]
d_excited_meas = np.array([10,10,10,10,10,10,10,10,10]) * 1e-3 # [kHz]


ground_meas = ground_meas_Nb
excited_meas = manu_ramsey_meas_Nb + ground_meas_Nb


def get_transitions_separated(e):
    # Correct the energy levels for the Lamb shift
    ground_transitions  = np.diff(e[:10]) 
    excited_transitions = np.diff(e[9:])
    return ground_transitions, excited_transitions

def get_log_likelihood_separated(hamiltonian: callable, excited: bool) -> callable:
    def log_likelihood_excited(x):
        h: Qobj = hamiltonian(x)
        _, excited_transitions = get_transitions_separated(h.eigenenergies())
        residuals = (excited_transitions - excited_meas ) / d_excited_meas
        return -0.5 * np.sum(residuals**2)

    def log_likelihood_ground(x):
        h: Qobj = hamiltonian(x)
        ground_transitions, _ = get_transitions_separated(h.eigenenergies())
        residuals = ( ground_transitions - ground_meas_Nb ) / d_ground_meas
        return -0.5 * np.sum(residuals**2)
    
    if excited:
        pass
        # return log_likelihood_excited
    else:
        return log_likelihood_ground

def get_q_tensor(D, E, Q, delta):
    delta -= np.pi/8
    c = E * np.cos(2 * delta)
    s = E * np.sin(2 * delta)
    q_tensor = np.array([
        [ -D/2 + c,        s, Q],
        [        s, -D/2 - c, 0],
        [        Q,        0, D]
    ])
    return q_tensor

def zeeman_hamiltonian_I(Bz) -> Qobj:
    return - Bz * (mu_Er * tensor(Sz, qeye(int(2*I+1))) + mu_Nb * tensor(qeye(2), Iz))

def quadrupole_hamiltonian_param(D, E, Q, delta) -> Qobj:
    q_tensor = get_q_tensor(D, E, Q, delta)
    h = 0
    for i, i1 in enumerate([Ix, Iy, Iz]):
        for j, i2 in enumerate([Ix, Iy, Iz]):
            h += q_tensor[i, j] * tensor(qeye(2), i1*i2)
    return h

# Define the Hamiltonian
def hamiltonian(x: np.ndarray) -> Qobj: 
    Bz, D, E, Q, delta  = x
    return zeeman_hamiltonian_I(Bz) + quadrupole_hamiltonian_param(D, E, Q, delta) 


def log_prior(x):
    Bz, D, E, Q, delta = x

    # -------- hard bounds (algebraic) --------------------------------
    if E <= 0.0 or Q <= 0.0:          # Q and E must be positive
        return -np.inf
    
    if delta < 0 or delta > np.pi/2:  # delta must be in [0, pi/2]
        return -np.inf
    
    return 0.0


def log_probability_excited(x):
    return log_prior(x) + get_log_likelihood_separated(hamiltonian,  excited=True)(x)

def log_probability_ground(x):
    return log_prior(x) + get_log_likelihood_separated(hamiltonian, excited=False)(x)

