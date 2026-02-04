from pdb import run
from unittest import case
import emcee
import corner
import pickle
from tqdm import tqdm
import numpy as np
from qutip import *
from multiprocess import Pool
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution, basinhopping
from IPython.display import display, Math
from scipy.stats import chi2
from protocolsClass import Hamiltonian_Fitter,State
import matplotlib.pyplot as plt
 

def main():
    ground_meas = np.array([7512786.3, 6847127.7, 6180056.7, 5510661.7, 4837199., 4156107.7, 3459061.2, 2730533.1, 1787541.6]) * 1e-3 # [kHz]
    manu_ramsey_meas = np.array([-134296, -133678, -132898, -131896, -130532, -128697, -125952, -124533, -88889]) * 1e-3 # [kHz] previous
    best_x_old =  [
            449.8217895516637,
            129.90322492463346,
            -237.05198836433365,
            -11.942209580809898,
            -149.3165679361477,
            1.5721261153027557,
            -0.8193213318825887,
            0.07781602296853966
        ]
    ground_meas_Nb = np.array([7560562.0 ,6894745.1 ,6227459.8 , 5557759.9 , 4883861.9 ,4202128.8 , 3504133.1, 2774604.8, 1822491.2]) * 1e-3
    manu_ramsey_meas_Nb = np.array([-136547.1, -135922.4, -135196.4, -134203.1, -132831.5, -130986.4, -128470.1, -122551.2, -128756.7])* 1e-3

    full_meas = np.concatenate((ground_meas_Nb,manu_ramsey_meas_Nb))
    full_meas_old = np.concatenate((ground_meas,manu_ramsey_meas))

    d_ground_meas = np.array([0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0001, 0.0002, 0.0002, 0.0002])# [kHz]
    d_manu_ramsey_meas = np.array([0.0450, 0.0298, 0.0236, 0.0285, 0.0223,  0.0182, 0.0159, 0.0195, 0.0175])# [kHz]
    d_full_meas = np.concatenate((d_ground_meas,d_manu_ramsey_meas))
    A_perp_meas = 51 # [kHz] | Measured through the Raman Rabi experiment
    A_simu = np.array([[-436.6,    0.,   -41.3],
                        [  -0.,  -448.4,    0. ],
                        [ -88.5,    0.,   129.8]])

    A_simu = np.array([[-441.66244757, -0.05970534,-6.00098845],[ -0.05970534, -441.66856909, 5.70123131],[ -6.00098845,5.70123131 ,131.30594568]])

    exp_id = '_Old_Nb_meas'
    exp_id = '_New_Nb_meas'

    fitter_ground = Hamiltonian_Fitter(ground_meas_Nb,d_ground_meas,State.Ground,id = exp_id)
    fitter_excited = Hamiltonian_Fitter(ground_meas_Nb + manu_ramsey_meas_Nb,d_manu_ramsey_meas,State.Excited,id = exp_id)


    fitter_full = Hamiltonian_Fitter(full_meas,d_full_meas,State.Full, meas_Aperp = A_perp_meas,simu_A= A_simu,id = exp_id)
    fitter_full.Load_results()
    fitter_full.Plot_Best()
    fitter_full.Print_values()
    fitter_full.chains()
    fitter_full.Plot_Quadropole()

if __name__ == "__main__":
    # not strictly required unless you freeze into an .exe, but harmless:
    from multiprocessing import freeze_support
    freeze_support()

    main()