# basic qutip imports
import numpy as np
from qutip import *
import matplotlib.pyplot as plt

# MCMC imports
import emcee
import corner
from tqdm import tqdm
from multiprocess import Pool


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




class Hamiltonian_Fitter():

    def __init__(self, excited_meas, ground_meas, d_excited_meas, d_ground_meas, excited):
        self.excited_meas = excited_meas
        self.ground_meas = ground_meas
        self.d_excited_meas = d_excited_meas
        self.d_ground_meas = d_ground_meas
        self.best_x = None
        self.median_x = None
        self.results = None

        
        # Lamb shift
        rel_electron_freq = np.cumsum([0, *self.excited_meas]) # [kHz]
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

    def get_log_likelihood_separated(self, hamiltonian: callable, excited: bool) -> callable:

        def log_likelihood_excited(x):
            h: Qobj = hamiltonian(x)
            _, excited_transitions = self.get_transitions_separated(h.eigenenergies())
            residuals = (excited_transitions - self.excited_meas) / self.d_excited_meas
            return -0.5 * np.sum(residuals**2)

        def log_likelihood_ground(x):
            h: Qobj = hamiltonian(x)
            ground_transitions, _ = self.get_transitions_separated(h.eigenenergies())
            residuals = (ground_transitions - self.ground_meas) / self.d_ground_meas
            return -0.5 * np.sum(residuals**2)

        return log_likelihood_excited if excited else log_likelihood_ground

    def plot_levels_and_residuals_separated(self, hamiltonian, x, title='', args={}):
        h: Qobj = hamiltonian(x, **args)
        ground_state, excited_state = self.get_transitions_separated(h.eigenenergies())

        fig, axs = plt.subplots(2, 2, figsize=(8, 6), tight_layout=True)
        plt.suptitle(title)

        plt.sca(axs[0, 0])
        plt.plot(self.excited_meas, 'o-', label='Measured')
        plt.plot(excited_state, 'o-', label='fit')
        plt.xlabel('Transition')
        plt.ylabel(r'$f_{excited}$ [kHz]')
        plt.legend()

        plt.sca(axs[0, 1])
        plt.errorbar(
            range(len(excited_state)),
            (self.excited_meas - excited_state) * 1e3,
            yerr=self.d_excited_meas * 1e3
        )
        plt.xlabel('Transition')
        plt.ylabel(r'res$( f_{excited})$ [Hz]')

        plt.sca(axs[1, 0])
        plt.plot(self.ground_meas, 'o-', label='Measured')
        plt.plot(ground_state, 'o-', label='fit')
        plt.legend()
        plt.xlabel('Transition')
        plt.ylabel(r'$f_{ground}$ [kHz]')

        plt.sca(axs[1, 1])
        plt.errorbar(
            range(len(self.ground_meas)),
            (self.ground_meas - ground_state) * 1e3,
            yerr=self.d_ground_meas * 1e3
        )
        plt.xlabel('Transition')
        plt.ylabel(r'res$( f_{ground}) $ [Hz]')

        plt.show()

    def run_MCMC(self, guess, excited, nwalkers=64, nsteps=10000):

        hamiltonian = self.hamiltonian
        log_likelihood = self.get_log_likelihood_separated(
            hamiltonian, excited=excited
        )

        pos = guess * (1 + 0.01 * np.random.randn(nwalkers, len(guess)))

        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, len(guess), log_likelihood, pool=pool)
            sampler.run_mcmc(pos, nsteps, progress=True)

        samples = sampler.get_chain(discard=500, flat=True)
        idx = sampler.get_log_prob()[500:].argmax()
        self.best_x = samples[idx]
        self.median_x = np.median(samples, axis=0)
        self.results = samples

        self.plot_levels_and_residuals_separated(
            self.hamiltonian, self.median_x,
            title='Median X errors'
        )
        print("median x : ",self.median_x)
        print("best x : ",self.best_x)
        return sampler

    def Plot_Best(self):
        self.plot_levels_and_residuals_separated(
            self.hamiltonian, self.best_x,
            title='Best X errors'
        )
    
    def Plot_Guess(self,guess):
        self.plot_levels_and_residuals_separated(
            self.hamiltonian, guess,
            title='Best X errors'
        )

    def Plot_corner(self):
        labels = ["Bz", "D", "E", "Q", "delta"]
        fig = corner.corner(self.results, labels=labels, truths=self.median_x)
        plt.show()


