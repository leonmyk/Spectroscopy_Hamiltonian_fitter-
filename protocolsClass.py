# basic qutip imports
import numpy as np
from qutip import *
import matplotlib.pyplot as plt

# MCMC imports
import emcee
import corner
from tqdm import tqdm
from multiprocess import Pool

from enum import Enum
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

class State(Enum):
    Ground = "ground"
    Excited = "excited"
    Full = "full"

manu_ramsey_meas = np.array([-134296, -133678, -132898, -131896, -130532, -128697, -125952, -124533, -88889]) * 1e-3 # [kHz]


class Hamiltonian_Fitter():

    def __init__(self, meas, d_meas, state:State, meas_Aperp:float = None , simu_A:float = None):

        self.state = state
        self.meas = meas
        self.d_meas = d_meas
        self.best_x = None
        self.median_x = None
        self.results = None
        self.meas_Aperp = meas_Aperp
        self.simu_A = simu_A

        
        # Lamb shift
        rel_electron_freq = np.cumsum([0, *manu_ramsey_meas]) # [kHz]
        g=5.24            # [kHz]
        kappa=700         # [kHz]
        self.lamb_shift_meas = rel_electron_freq * g**2 / (kappa**2/4 + rel_electron_freq**2) # [kHz]

    def get_q_tensor(self, D, E, Q, delta):
        c = E * np.cos(2 * delta)
        s = E * np.sin(2 * delta)
        q_tensor = np.array([
            [-D/2 + c,  s, Q],
            [s, -D/2 - c, 0],
            [Q, 0, D]
        ])
        return q_tensor

    def zeeman_hamiltonian(self, Bz) -> Qobj:
        return -Bz * (
            mu_Er * tensor(Sz, qeye(int(2*I+1))) +
            mu_Nb * tensor(qeye(2), Iz)
        )

    def quadrupole_hamiltonian_param(self, D, E, Q, delta) -> Qobj:
        q_tensor = self.get_q_tensor(D, E, Q, delta)
        h = 0
        for i, i1 in enumerate([Ix, Iy, Iz]):
            for j, i2 in enumerate([Ix, Iy, Iz]):
                h += q_tensor[i, j] * tensor(qeye(2), i1 * i2)
        return h

    def hamiltonian(self, x: np.ndarray) -> Qobj:
        Bz, D, E, Q2, Q3 = x
        return (
            self.zeeman_hamiltonian(Bz) +
            self.quadrupole_hamiltonian_param(D, E, Q2, Q3)
        )

    def get_transitions_separated(self, e):
        ground_transitions = np.diff(e[:10])
        excited_transitions = np.diff(e[10:] + self.lamb_shift_meas)
        return ground_transitions, excited_transitions

    def get_log_likelihood_separated(self,x):


        if self.state == State.Excited :
            h: Qobj = self.hamiltonian(x)
            ground_transitions, excited_transitions = self.get_transitions_separated(h.eigenenergies())
            residuals = (excited_transitions - self.meas) / self.d_meas

        elif self.state == State.Ground :
            h: Qobj = self.hamiltonian(x)
            ground_transitions, excited_transitions = self.get_transitions_separated(h.eigenenergies())
            residuals = (ground_transitions - self.meas) / self.d_meas
        else :
            h: Qobj = self.Full_hamiltonian(x)
            ground_transitions, excited_transitions = self.get_transitions_separated(h.eigenenergies())
            residuals = (np.concatenate((ground_transitions,excited_transitions - ground_transitions)) - self.meas) / self.d_meas

        return -0.5 * np.sum(residuals**2)

    
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

    def zeeman_full_hamiltonian(self, Bz) -> Qobj:
        return - Bz * (mu_Er * tensor(Sz, qeye(int(2*I+1))) + mu_Nb * tensor(qeye(2), Iz))

    def hyperfine_hamiltonian(self,A) -> Qobj:
        h = 0 # Hyperfine interaction 
        for i, s_op in enumerate([Sx, Sy]):
            for j, i_op in enumerate([Ix, Iy, Iz]):
                h += self.simu_A[i, j] * tensor(s_op, i_op)
        return A * tensor(Sz, Iz) + self.meas_Aperp * tensor(Sz, Ix) + h

    def full_quadrupole_hamiltonian_param(self,D, S1, S2, delta, theta) -> Qobj:
        q_tensor = self.get_full_q_tensor(D, S1, S2, delta, theta)
        h = 0
        for i, i1 in enumerate([Ix, Iy, Iz]):
            for j, i2 in enumerate([Ix, Iy, Iz]):
                h += q_tensor[i, j] * tensor(qeye(2), i1*i2)
        return h

    def sdq_hamiltonian_param(self,Dz) -> Qobj:    
        q_tensor = self.get_full_q_tensor(Dz, 0,0,0,0)
        h = 0
        for i, i1 in enumerate([Ix, Iy, Iz]):
            for j, i2 in enumerate([Ix, Iy, Iz]):
                h += q_tensor[i, j] * tensor(Sz, i1*i2)
        return h

    def hexadecapole_hamiltonian(self,Hx) -> Qobj:
        # Hexadecapole term is not implemented in this context, but can be added similarly
        return Hx * tensor(Sz, Iz*Iz*Iz*Iz)

    # Define the Hamiltonian
    def Full_hamiltonian(self,x: np.ndarray) -> Qobj: 
        Bz, A, D, S1, S2, delta, alpha, Dz = x
        return self.zeeman_full_hamiltonian(Bz) +\
            self.hyperfine_hamiltonian(A) +\
            self.full_quadrupole_hamiltonian_param(D, S1, S2, delta, alpha) +\
            self.sdq_hamiltonian_param(Dz) #+\
            #hexadecapole_hamiltonian(Hx)

    def Plot_full(self, x, title='Full Fit'):

        h: Qobj = self.Full_hamiltonian(x)
        ground_transitions, excited_transitions = self.get_transitions_separated(h.eigenenergies())
        fit = np.concatenate((ground_transitions,excited_transitions))
        error = (np.concatenate((ground_transitions,excited_transitions - ground_transitions)) - self.meas)
        meas_to_plot = self.meas + np.concatenate((np.zeros(len(ground_transitions)),ground_transitions))
        
        fig, axs = plt.subplots(3, 1, figsize=(8, 6), tight_layout=True,sharex=True)
        plt.suptitle(title)
        plt.sca(axs[0])
        plt.plot(meas_to_plot[:9], 'o', marker = 'v', label= r"$\omega^{\downarrow}_{{n(n+1)}/2\pi}$", color = 'orange')
        plt.plot(meas_to_plot[9:], 'o', marker = '^', label= r"$\omega^{\uparrow}_{{n(n+1)}/2\pi}$", color = 'blue')
        # plt.plot(fit, 'o-', label='fit')
        plt.xlabel('Transition')
        plt.ylabel(rf'$f_{self.state.value}$ [kHz]')
        plt.legend()

        plt.sca(axs[1])
        plt.errorbar(
            range(len(fit[:9])),
            (error[:9]) * 1e3,fmt = 'o',
            yerr=self.d_meas[:9] * 1e3,
            marker = 'v', color = 'orange'
        )
        plt.ylabel(r'$residual_{{\downarrow}} [Hz]$')

        plt.sca(axs[2])
        plt.errorbar(
            range(len(fit[:9])),
            (error[9:]) * 1e3,fmt = 'o',
            yerr=self.d_meas[9:] * 1e3,
            marker = '^', color = 'blue'
        )
        plt.xlabel('Transition')
        plt.ylabel(r'$residual_{{\uparrow}} [Hz]$')

        
        plt.show()
        
    def plot_levels_and_residuals_separated(self, x, title='',args={}):

        if self.state == State.Excited :
            h: Qobj = self.hamiltonian(x)
            ground_transitions, excited_transitions = self.get_transitions_separated(h.eigenenergies())
            fit = excited_transitions 
            error = (excited_transitions - self.meas)
            meas_to_plot = self.meas


        elif self.state == State.Ground :
            h: Qobj = self.hamiltonian(x)
            ground_transitions, excited_transitions = self.get_transitions_separated(h.eigenenergies())
            fit = ground_transitions
            error = (ground_transitions - self.meas)
            meas_to_plot = self.meas

        else :
            h: Qobj = self.Full_hamiltonian(x)
            ground_transitions, excited_transitions = self.get_transitions_separated(h.eigenenergies())
            fit = np.concatenate((ground_transitions,excited_transitions))
            error = (np.concatenate((ground_transitions,excited_transitions - ground_transitions)) - self.meas)
            meas_to_plot = self.meas + np.concatenate((np.zeros(len(ground_transitions)),ground_transitions))

        
        
        fig, axs = plt.subplots(1, 2, figsize=(8, 6), tight_layout=True)
        plt.suptitle(title)

        plt.sca(axs[0])
        plt.plot(meas_to_plot, 'o-', label='Measured')
        plt.plot(fit, 'o-', label='fit')
        plt.xlabel('Transition')
        plt.ylabel(rf'$f_{self.state.value}$ [kHz]')
        plt.legend()

        plt.sca(axs[1])
        plt.errorbar(
            range(len(fit)),
            (error) * 1e3,
            yerr=self.d_meas * 1e3
        )
        plt.xlabel('Transition')
        plt.ylabel(rf'res$( f_{self.state.value})$ [Hz]')

        plt.show()

    def run_MCMC(self, guess,nwalkers=64, nsteps=10000, var = 0.01):

        if self.state == State.Full :
            log_likelihood = self.get_log_likelihood_separated  
        else :
            log_likelihood = self.get_log_likelihood_separated

        pos = guess * (1 +  var * np.random.randn(nwalkers, len(guess)))

        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, len(guess), log_likelihood, pool=pool)
            sampler.run_mcmc(pos, nsteps, progress=True)

        samples = sampler.get_chain(discard=500, flat=True)
        idx = sampler.get_log_prob()[500:].argmax()
        self.best_x = samples[idx]
        self.median_x = np.median(samples, axis=0)
        self.results = samples

        print("median x : ",self.median_x)
        print("best x : ",self.best_x)

        self.plot_levels_and_residuals_separated(
            self.median_x,
            title='Median X errors'
        )

        return sampler

    def Plot_Best(self):
        self.plot_levels_and_residuals_separated(
            self.best_x,
            title='Best X errors'
        )
    
    def Plot_Guess(self,guess):
        self.plot_levels_and_residuals_separated(
            guess,
            title='Best X errors'
        )

    def Plot_corner(self):
        if self.state == State.Full:
            labels = ["Bz", "A", "D", "S1", "S2", "delta", "alpha", "Dz"]
        else :
            labels = ["Bz", "D", "E", "Q", "delta"]
        fig = corner.corner(self.results, labels=labels, truths=self.median_x)
        plt.show()


