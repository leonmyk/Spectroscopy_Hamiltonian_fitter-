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

from functions import Full_hamiltonian
from functions import hamiltonian
from functions import hamiltonian_Heca
from functions import get_q_tensor
from functions import normalise_Histogram_Height
from functions import pretty_mcmc
from functions import get_full_q_tensor

# Constants
mu_Nb = 10.4213  # [kHz / mT]
mu_Er = - 17_350 # [kHz / mT]

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

class State(Enum):
    Ground = "ground"
    Excited = "excited"
    Full = "full"
    Heca = "heca"



class Hamiltonian_Fitter():

    def __init__(self, meas, std_meas, state:State,id:str = '00', meas_Aperp:float = None , simu_A:float = None):

        self.state = state
        self.meas = meas
        self.std_meas = std_meas
        self.best_x = {}
        self.median_x = {}
        self.results = {}
        self.sampler = None
        self.meas_Aperp = meas_Aperp
        self.simu_A = simu_A
        self.id = id

        
        # Lamb shift
        rel_electron_freq = np.cumsum([0, *meas]) # [kHz]
        g=5.24            # [kHz]
        kappa=700         # [kHz]
        self.lamb_shift_meas = rel_electron_freq * g**2 / (kappa**2/4 + rel_electron_freq**2) # [kHz]


        if self.state == State.Full:
            self.labels = ["Bz", "A", "D", "S1", "S2", "delta", "alpha", "Dz"]
        else :
            self.labels = ["Bz", "D", "E", "F", "delta"]


    def log_prior_full(self,x):
        Bz, A, D, S1, S2, delta, alpha, Dz = x

        # # -------- hard bounds (algebraic) --------------------------------
        # if S1 <= 0.0 or S2 <= 0.0:          # S1 and S2 must be positive
        #     return -np.inf
        
        # delta and alpha must be in (-π/2, π/2]
        # if not (-np.pi/2 < delta <= np.pi/2 and -np.pi/2 < alpha <= np.pi/2):
        #     return -np.inf
        
        return 0.0

    def log_prior(self,x):
        Bz, D, E, Q, delta = x

        # -------- hard bounds (algebraic) --------------------------------
        if Q <= 0.0 or not (-0.5 < delta <= 3):          # Q and E must be positive
            return -np.inf

        
        return 0.0

    def get_transitions_separated(self, e):
        ground_transitions = np.diff(e[:8])
        excited_transitions = np.diff(e[8:] + self.lamb_shift_meas)
        return ground_transitions, excited_transitions

    def get_log_likelihood_separated(self,x):


        if self.state == State.Excited :
            h: Qobj = hamiltonian(x)
            ground_transitions, excited_transitions = self.get_transitions_separated(h.eigenenergies())
            residuals = (excited_transitions - self.meas) / self.std_meas + self.log_prior(x)

        elif self.state == State.Ground :
            h: Qobj = hamiltonian(x)
            ground_transitions, excited_transitions = self.get_transitions_separated(h.eigenenergies())
            residuals = (ground_transitions - self.meas) / self.std_meas + self.log_prior(x)
            
        elif self.state == State.Full :
            h: Qobj = Full_hamiltonian(x)
            ground_transitions, excited_transitions = self.get_transitions_separated(h.eigenenergies())
            meas_to_compare = np.concatenate((self.meas[:9],self.meas[9:] + self.meas[:9]))
            residuals = (np.concatenate((ground_transitions,excited_transitions)) - meas_to_compare) / self.std_meas

        elif self.state == State.Heca :
            h: Qobj = hamiltonian_Heca(x)
            ground_transitions, _ = self.get_transitions_separated(h.eigenenergies())
            jaime_energies = np.dif(ground_transitions)
            meas_to_compare = np.concatenate((self.meas[:9],self.meas[9:] + self.meas[:9]))
            residuals = (jaime_energies- self.meas) / self.std_meas
        

        residuals_sum = -0.5 * np.sum(residuals**2)+ self.log_prior_full(x) if self.state == State.Full else -0.5 * np.sum(residuals**2)+ self.log_prior(x)
        return residuals_sum

    def Get_deriv(self, offset, guess=None, indices_to_plot=[0]):

        if guess == None:
            guess = self.best_x[self.state.value]

        n_points = 300
        x_list1 = np.array(np.copy(guess))
        x_list2 = np.array(np.copy(guess))
        
        
        x_list1[0] = x_list1[0] - offset
        x_list2[0] = x_list2[0] + offset

        h1 = Full_hamiltonian(x_list1) if self.state == State.Full else hamiltonian(x_list1)
        h2 = Full_hamiltonian(x_list2) if self.state == State.Full else hamiltonian(x_list2)
        
        energies1 = np.concatenate(self.get_transitions_separated(h1.eigenenergies()))
        energies2 = np.concatenate(self.get_transitions_separated(h2.eigenenergies()))

        gradients = (energies2-energies1)/(2*offset)


        fig, ax = plt.subplots(2, figsize=(8, 20), sharex=True)
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        w_center = (energies1 + energies2)/2

        label_e = ''
        label_d = ''
        
        for i in indices_to_plot :
        
            label_e += (
                "\n"
                f"Transition {i}  "
                f"(w = {w_center[i]:.2f} kHz)"
            )
            
            label_d += (
                "\n"
                f"Transition {i}  "
                f"(dE/dB = {gradients[i]:.2f} kHz/mT)"
            )

        ax[0].plot(indices_to_plot,w_center[indices_to_plot],'o', label=label_e)
        ax[1].plot(indices_to_plot,gradients[indices_to_plot],'o', label=label_d)
            

        # Axis labels
        ax[1].set_xlabel("transition")
        ax[0].set_ylabel("Energy (kHz)")
        ax[1].set_ylabel("dE / dB (kHz / mT)")

        # Titles
        ax[0].set_title("Transition energies")
        ax[1].set_title("Energy derivatives")

        # Legends
        ax[0].legend()
        ax[1].legend()
        # ax[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
        # ax[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)


        # Global title
        fig.suptitle(
            f"Spectroscopy around B = {guess[0]:.4f} mT using " + self.state.value + " hamiltonian",
            fontsize=12
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def Plot_Quadropole(self,title='Quadropole Tensor Elements Distribution'):

        fig,axs = plt.subplots(1,3, figsize=(16,12), tight_layout=True,sharey=True)
        plt.suptitle(title)
        Q1 = {}
        Q2 = {}
        Q3 = {}

        for state in [State.Excited,State.Ground]:

            Q1[state.value] = []
            Q2[state.value] = []
            Q3[state.value] = []

            for res in self.results[state.value]:

                if state == State.Full :
                    q_gr_f = get_full_q_tensor(res[2] - res[7]/2, res[3], res[4], res[5], res[6])
                    q_ex_f = get_full_q_tensor(res[2] + res[7]/2, res[3], res[4], res[5], res[6])
                    vals_gr_f, vals_ex_f = np.linalg.eigvals(q_gr_f), np.linalg.eigvals(q_ex_f)
                    Qz_gr,Qy_gr,Qx_gr = np.sort(vals_gr_f)
                    Qz_ex,Qy_ex,Qx_ex = np.sort(vals_ex_f)
                    Q3[state.value].append((Qz_gr, Qz_ex))
                    Q2[state.value].append((Qy_gr, Qy_ex))
                    Q1[state.value].append((Qx_gr, Qx_ex))

                else :
                    D, E, Q, delta = res[1], res[2], res[3], res[4]
                    q_tensor = get_q_tensor(D, E, Q, delta)
                    Qz, Qy, Qx = np.linalg.eigvalsh(q_tensor)
                    Q3[state.value].append(Qz)
                    Q2[state.value].append(Qy)
                    Q1[state.value].append(Qx)

        sorted_index = np.argsort([np.mean(Q3[State.Ground.value]), np.mean(Q2[State.Ground.value]), np.mean(Q1[State.Ground.value])])
        QZZ, QYY, QXX = [[Q3, Q2, Q1][i] for i in sorted_index[::-1]]

        values = pretty_mcmc(np.array([np.array(QZZ[State.Ground.value]), np.array(QYY[State.Ground.value]), np.array(QXX[State.Ground.value])]), sig_figs=2)

        QXX_ex_offsets = (QXX[State.Excited.value]-np.mean(QXX[State.Ground.value]))
        QXX_gd_offsets = (QXX[State.Ground.value]-np.mean(QXX[State.Ground.value]))

        (e_g,c_g,w1),(e_e,c_e,w2) = normalise_Histogram_Height(QXX_gd_offsets,QXX_ex_offsets,20,120)
        axs[0].bar(e_g, c_g, width=w1, align="edge", alpha=0.4, label="Ground")
        axs[0].bar(e_e, c_e, width=w2, align="edge", alpha=0.4, label="Excited")
        axs[0].set_xlabel(r'$Q_{XX} (Hz)$')
        axs[0].set_title(f"{values[2][1]} KHz (-{values[2][0]*1e3} Hz / +{values[2][2]*1e3} Hz)")
        # axs[0].set_xlim(right=100,left=-100)
        axs[0].set_ylabel('Normalized counts')

        QYY_ex_offsets = (QYY[State.Excited.value]-np.mean(QYY[State.Ground.value]))
        QYY_gd_offsets = (QYY[State.Ground.value]-np.mean(QYY[State.Ground.value]))
        (e_g,c_g,w1),(e_e,c_e,w2) = normalise_Histogram_Height(QYY_gd_offsets*1e3,QYY_ex_offsets*1e3,20,120)
        axs[1].bar(e_g, c_g, width=w1, align="edge", alpha=0.4, label="Ground")
        axs[1].bar(e_e, c_e, width=w2, align="edge", alpha=0.4, label="Excited")
        axs[1].set_xlabel(r'$Q_{YY} (Hz)$')
        axs[1].set_title(f"{values[1][1]} KHz (-{values[1][0]*1e3} Hz / +{values[1][2]*1e3} Hz)")   
        # axs[1].set_xlim(right=500,left=-500)
 
 
 
        QZZ_ex_offsets = (QZZ[State.Excited.value]-np.mean(QZZ[State.Ground.value]))
        QZZ_gd_offsets = (QZZ[State.Ground.value]-np.mean(QZZ[State.Ground.value]))
        (e_g,c_g,w1),(e_e,c_e,w2) = normalise_Histogram_Height(QZZ_gd_offsets*1e3,QZZ_ex_offsets*1e3,20,120)
        axs[2].bar(e_g, c_g, width=w1, align="edge", alpha=0.4, label="Ground")
        axs[2].bar(e_e, c_e, width=w2, align="edge", alpha=0.4, label="Excited")
        axs[2].set_xlabel(r'$Q_{ZZ} (Hz)$')
        axs[2].set_title(f"{values[0][1]} KHz (-{values[0][0]*1e3} Hz / +{values[0][2]*1e3} Hz)")
        # axs[2].set_xlim(right=500,left=-500)

        axs[0].legend()
        axs[1].legend()
        axs[2].legend()


        plt.show()

    def Plot_full(self, x, title='Full Fit'):

        h: Qobj = Full_hamiltonian(x)
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
            yerr=self.std_meas[:9] * 1e3*2,
            marker = 'v', color = 'orange'
        )
        plt.ylabel(r'$residual_{{\downarrow}} [Hz]$')

        plt.sca(axs[2])
        plt.errorbar(
            range(len(fit[:9])),
            (error[9:]) * 1e3,fmt = 'o',
            yerr=self.std_meas[9:] * 1e3*2,
            marker = '^', color = 'blue'
        )
        plt.xlabel('Transition')
        plt.ylabel(r'$residual_{{\uparrow}} [Hz]$')

        
        plt.show()
        
    def plot_levels_and_residuals_separated(self, x, title='',args={}):

        if self.state == State.Excited :
            h: Qobj = hamiltonian(x)
            ground_transitions, excited_transitions = self.get_transitions_separated(h.eigenenergies())
            fit = excited_transitions 
            error = (excited_transitions - self.meas)
            meas_to_plot = self.meas


        elif self.state == State.Ground :
            h: Qobj = hamiltonian(x)
            ground_transitions, excited_transitions = self.get_transitions_separated(h.eigenenergies())
            fit = ground_transitions
            error = (ground_transitions - self.meas)
            meas_to_plot = self.meas

        else :
            h: Qobj = Full_hamiltonian(x)
            ground_transitions, excited_transitions = self.get_transitions_separated(h.eigenenergies())
            fit = np.concatenate((ground_transitions,excited_transitions))
            error = (np.concatenate((ground_transitions,excited_transitions - ground_transitions)) - self.meas)
            meas_to_plot = self.meas + np.concatenate((np.zeros(len(ground_transitions)),self.meas[:9]))

        
        
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
            error * 1e3,
            yerr=self.std_meas * 1e3 * 2,
            fmt='x',              # cross marker
            color='black',        # marker color
            ecolor='red',         # error bar color
            linestyle='none'      # no connecting line
        )
        plt.xlabel('Transition')
        plt.ylabel(rf'res$( f_{self.state.value})$ [Hz]')

        plt.show()
        
    def chains(self):
        # Get the chains from the sampler
        
        ndim = self.results[self.state.value].shape[1]

        fig, axes = plt.subplots(ndim, 1, figsize=(10, 2 * ndim), tight_layout=True, sharex=True)
        for i in range(ndim):
            ax = axes[i]
            ax.plot(self.results[self.state.value][:,i], alpha=0.5,)  # Plot all walkers for parameter i
            ax.set_ylabel(self.labels[i])

        plt.show()

    def run_MCMC(self, guess,nwalkers=64, nsteps=10000, var = 0.01):
        

        if self.state == State.Full :
            log_likelihood = self.get_log_likelihood_separated  
        else :
            log_likelihood = self.get_log_likelihood_separated

        pos = guess * (1 +  var * np.random.randn(nwalkers, len(guess)))
        
        self.results = {}
        self.sampler = None
        
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


        values = pretty_mcmc(self.results[self.state.value], sig_figs=2)
        for i in range(len(self.labels)):
            low, central, high = values[i]
            txt = (rf"\mathrm{{{self.labels[i]}}}"
               rf" = {central}_{{-{low}}}^{{+{high}}}")
            display(Math(txt))

        
        self.plot_levels_and_residuals_separated(
            self.median_x[self.state.value],
            title='Median X errors'
        )

        return sampler
    
    def Print_values(self):
        values = pretty_mcmc(self.results[self.state.value], sig_figs=2)
        for i in range(len(self.labels)):
            low, central, high = values[i]
            txt = (rf"\mathrm{{{self.labels[i]}}}"
               rf" = {central}_{{-{low}}}^{{+{high}}}")
            display(Math(txt))

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
        residuals_avg = np.average(np.abs(self.get_log_likelihood_separated(self.best_x[self.state.value])))
        self.plot_levels_and_residuals_separated(
            self.best_x[self.state.value],
            title= rf'Best X errors average residual is {residuals_avg}'
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


ground_meas_old_Nb = np.array([7512786.3, 6847127.7, 6180056.7, 5510661.7, 4837199., 4156107.7, 3459061.2, 2730533.1, 1787541.6]) * 1e-3 # [kHz]
manu_ramsey_meas_old_Nb = np.array([-134296, -133678, -132898, -131896, -130532, -128697, -125952, -124533, -88889]) * 1e-3 # [kHz] previous
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

ground_meas_Ca = np.array([997843.4 ,1090793.4 ,1185442.1 ,1281215.1 ,1377750.3 ,1474814.6 ,1572254.3])*1e-3
manu_ramsey_meas_Ca = np.array([-27703.3,-27670.2,-27594.3,-27518.5,-27449.3,-27376.6,-27349.6])*1e-3

d_ground_meas_Ca = np.array([0.0273, 0.0249, 0.0277, 0.0283, 0.0286, 0.0409, 0.0654])*1e-3# [kHz]
d_manu_ramsey_meas_Ca = np.array([5.7657,5.5012, 5.8450, 5.3050, 6.7760, 8.1508, 13.7237])*1e-3# [kHz]
full_meas_Ca = np.concatenate((ground_meas_Ca,manu_ramsey_meas_Ca))
d_full_meas_Ca = np.concatenate((d_ground_meas_Ca,d_manu_ramsey_meas_Ca))



full_meas_Nb = np.concatenate((ground_meas_Nb,manu_ramsey_meas_Nb))
full_meas_old_Nb = np.concatenate((ground_meas_old_Nb,manu_ramsey_meas_old_Nb))

d_ground_meas_Nb = np.array([0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0001, 0.0002, 0.0002, 0.0002])# [kHz]
d_manu_ramsey_meas_Nb = np.array([0.0450, 0.0298, 0.0236, 0.0285, 0.0223,  0.0182, 0.0159, 0.0195, 0.0175])# [kHz]
d_full_meas_Nb = np.concatenate((d_ground_meas_Nb,d_manu_ramsey_meas_Nb))
A_perp_meas_Nb = 55 # [kHz] | Measured through the Raman Rabi experiment
A_perp_meas_Ca = 20 # [kHz] | Measured through the Raman Rabi experiment
A_simu_Nb = np.array([[-436.6,    0.,   -41.3],
                    [  -0.,  -448.4,    0. ],
                    [ -88.5,    0.,   129.8]])

A_simu_Nb = 1e3*np.array([[ 0.43899564,  0.,          0.02076148],
                    [-0.,          0.44172529, -0.        ],
                    [-0.04857903,  0.,          0.12994526]])


exp_id = '_Ca_meas'

fitter_ground = Hamiltonian_Fitter(ground_meas_Ca,d_ground_meas_Ca,State.Ground,id = exp_id)
fitter_excited = Hamiltonian_Fitter(ground_meas_Ca + d_manu_ramsey_meas_Ca,d_manu_ramsey_meas_Ca,State.Excited,id = exp_id)
fitter_full = Hamiltonian_Fitter(full_meas_Ca,d_full_meas_Ca,State.Full, meas_Aperp = A_perp_meas_Ca,simu_A= A_simu_Nb,id = exp_id)
fitter_full.Load_results()
# fitter_ground.Load_results()
# fitter_ground.Plot_Best()
fitter_full.Plot_Quadropole()
# fitter_full.plot_levels_and_residuals_separated(fitter_full.median_x[State.Ground.value], title='Median X errors')


