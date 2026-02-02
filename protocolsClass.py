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

class State(Enum):
    Ground = "ground"
    Excited = "excited"
    Full = "full"

manu_ramsey_meas = np.array([-134296, -133678, -132898, -131896, -130532, -128697, -125952, -124533, -88889]) * 1e-3 # [kHz]


class Hamiltonian_Fitter():

    def __init__(self, meas, d_meas, state:State,id:str = '00', meas_Aperp:float = None , simu_A:float = None):

        self.state = state
        self.meas = meas
        self.d_meas = d_meas
        self.best_x = {}
        self.median_x = {}
        self.results = {}
        self.meas_Aperp = meas_Aperp
        self.simu_A = simu_A
        self.id = id

        
        # Lamb shift
        rel_electron_freq = np.cumsum([0, *manu_ramsey_meas]) # [kHz]
        g=5.24            # [kHz]
        kappa=700         # [kHz]
        self.lamb_shift_meas = rel_electron_freq * g**2 / (kappa**2/4 + rel_electron_freq**2) # [kHz]


        # self.Load_results()

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
        Bz, D, E, Q, delta = x
        return (
            self.zeeman_hamiltonian(Bz) +
            self.quadrupole_hamiltonian_param(D, E, Q, delta)
        )
        
    def log_prior_full(self,x):
        Bz, A, D, S1, S2, delta, alpha, Dz = x

        # -------- hard bounds (algebraic) --------------------------------
        if S1 <= 0.0 or S2 <= 0.0:          # S1 and S2 must be positive
            return -np.inf
        
        # delta and alpha must be in (-π/2, π/2]
        if not (0 < delta <= np.pi/2 and 0 < alpha <= np.pi/2):
            return -np.inf
        
        return 0.0

    def log_prior(self,x):
        Bz, D, E, Q, delta = x

        # -------- hard bounds (algebraic) --------------------------------
        if Q <= 0.0 or not (-0.5 < delta <= 3):          # Q and E must be positive
            return -np.inf

        
        return 0.0

    def get_transitions_separated(self, e):
        ground_transitions = np.diff(e[:10])
        excited_transitions = np.diff(e[10:] + self.lamb_shift_meas)
        return ground_transitions, excited_transitions

    def get_log_likelihood_separated(self,x):


        if self.state == State.Excited :
            h: Qobj = self.hamiltonian(x)
            ground_transitions, excited_transitions = self.get_transitions_separated(h.eigenenergies())
            residuals = (excited_transitions - self.meas) / self.d_meas + self.log_prior(x)

        elif self.state == State.Ground :
            h: Qobj = self.hamiltonian(x)
            ground_transitions, excited_transitions = self.get_transitions_separated(h.eigenenergies())
            residuals = (ground_transitions - self.meas) / self.d_meas + self.log_prior(x)
        else :
            h: Qobj = self.Full_hamiltonian(x)
            ground_transitions, excited_transitions = self.get_transitions_separated(h.eigenenergies())
            meas_to_compare = np.concatenate((self.meas[:9],self.meas[9:] + self.meas[:9]))
            residuals = (np.concatenate((ground_transitions,excited_transitions)) - meas_to_compare) / self.d_meas

        residuals_sum = -0.5 * np.sum(residuals**2)+ self.log_prior_full(x) if self.state == State.Full else -0.5 * np.sum(residuals**2)+ self.log_prior(x)
        return residuals_sum

    
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


    def Plot_Quadropole(self):

        fig,axs = plt.subplots(1,3, figsize=(8,6), tight_layout=True)
        plt.suptitle('Quadropole Tensor Elements Distribution')
        QXX = {}
        QYY = {}
        QZZ = {}

        for state in State:

            QXX[state.value] = []
            QYY[state.value] = []
            QZZ[state.value] = []

            for res in self.results[state.value]:

                if state == State.Full :
                    q_gr_f = self.get_full_q_tensor(res[2] - res[7]/2, res[3], res[4], res[5], res[6])
                    q_ex_f = self.get_full_q_tensor(res[2] + res[7]/2, res[3], res[4], res[5], res[6])
                    vals_gr_f, vals_ex_f = np.linalg.eigvals(q_gr_f), np.linalg.eigvals(q_ex_f)
                    Qx_gr,Qy_gr,Qz_gr = np.sort(vals_gr_f)
                    Qx_ex,Qy_ex,Qz_ex = np.sort(vals_ex_f)
                    QXX[state.value].append((Qx_gr, Qx_ex))
                    QYY[state.value].append((Qy_gr, Qy_ex))
                    QZZ[state.value].append((Qz_gr, Qz_ex))

                else :
                    D, E, Q, delta = res[1], res[2], res[3], res[4]
                    q_tensor = self.get_q_tensor(D, E, Q, delta)
                    Qx,Qy,Qz = np.linalg.eigvalsh(q_tensor)

                    QXX[state.value].append(Qx)
                    QYY[state.value].append(Qy)
                    QZZ[state.value].append(Qz)


        axs[0].hist(QXX[State.Excited.value]-np.mean(QXX[State.Ground.value]), bins=20, alpha=0.5, label='Excited')
        axs[0].hist(QXX[State.Ground.value]-np.mean(QXX[State.Ground.value]), bins=20, alpha=0.5, label='Ground')
        axs[0].set_xlabel(r'$Q_{XX}$')
        axs[0].set_title(f'{np.mean(QXX[State.Ground.value])}')
        # axs[0].set_xlim(right=1,left=-1)

        axs[1].hist(QYY[State.Excited.value]-np.mean(QYY[State.Ground.value]), bins=20, alpha=0.5, label='Excited')
        axs[1].hist(QYY[State.Ground.value]-np.mean(QYY[State.Ground.value]), bins=20, alpha=0.5, label='Ground')
        axs[1].set_xlabel(r'$Q_{YY}$')
        axs[1].set_title(f'{np.mean(QYY[State.Ground.value])}')
        # axs[1].set_xlim(right=1,left=-1)

        axs[2].hist(QZZ[State.Excited.value]-np.mean(QZZ[State.Ground.value]), bins=20, alpha=0.5, label='Excited')
        axs[2].hist(QZZ[State.Ground.value]-np.mean(QZZ[State.Ground.value]), bins=20, alpha=0.5, label='Ground')
        axs[2].set_xlabel(r'$Q_{ZZ}$')
        axs[2].set_title(f'{np.mean(QZZ[State.Ground.value])}')
        # axs[2].set_xlim(right=1,left=-1)

        axs[0].legend()
        axs[1].legend()
        axs[2].legend()

        print("QYY[State.Excited.value]:", QYY[State.Excited.value])

        plt.show()


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
        plt.plot(meas_to_plot, 'o', label='Measured')
        plt.plot(fit, 'o', label='fit')
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

        self.best_x[self.state.value] = samples[idx]
        self.median_x[self.state.value] = np.median(samples, axis=0)
        self.results[self.state.value] = samples

        print("median x : ",self.median_x[self.state.value])
        print("best x : ",self.best_x[self.state.value])

        self.plot_levels_and_residuals_separated(
            self.median_x[self.state.value],
            title='Median X errors'
        )

        return sampler

    def Save_results(self):
        
        filename = f'mcmc_results_{self.state.value}' + self.id + '.json'
        with open(filename, "w", encoding="utf-8") as f:
            json.dump({
                "best_x": self.best_x[self.state.value].tolist(),
                "median_x": self.median_x[self.state.value].tolist(),
                "results": self.results[self.state.value].tolist()
            }   , f, indent=4, ensure_ascii=False)

    def Load_results(self):

        filename = f'mcmc_results_{State.Ground.value}' + self.id + '.json'
        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.best_x[State.Ground.value] = np.array(data["best_x"])
                self.median_x[State.Ground.value] = np.array(data["median_x"])
                self.results[State.Ground.value] = np.array(data["results"])
        except FileNotFoundError:
            print(f"File {filename} not found. Skipping loading ground state results.")
        
        filename = f'mcmc_results_{State.Excited.value}' + self.id + '.json'
        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.best_x[State.Excited.value] = np.array(data["best_x"])
            self.median_x[State.Excited.value] = np.array(data["median_x"])
            self.results[State.Excited.value] = np.array(data["results"])
        except FileNotFoundError:
            print(f"File {filename} not found. Skipping loading excited state results.")

        filename = f'mcmc_results_{State.Full.value}' + self.id + '.json'
        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.best_x[State.Full.value] = np.array(data["best_x"])
                self.median_x[State.Full.value] = np.array(data["median_x"])
            self.results[State.Full.value] = np.array(data["results"])
        except FileNotFoundError:
            print(f"File {filename} not found. Skipping loading full state results.")

    def Plot_Best(self):
        self.plot_levels_and_residuals_separated(
            self.best_x[self.state.value],
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
        fig = corner.corner(self.results[self.state.value], labels=labels, truths=self.median_x[self.state.value])
        plt.show()


