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
mu_Nb = 10.4213  # [kHz / mT]

# Calcium-43 nuclear magnetic moment
gamma_Ca_ref = -2.86899e6          # Hz/T  (=-2.86899 MHz/T)  (43Ca)  :contentReference[oaicite:3]{index=3}
mu_Ca = gamma_Ca_ref * h           # J/T
g_Ca  = mu_Ca / mu_N   
mu_Ca = - 2.87


class SpinSystem:

    def __init__(self, S=1/2, I=7/2):
        self.S = S
        self.I = I

        # Electron operators
        self.Sx = jmat(S, 'x')
        self.Sy = jmat(S, 'y')
        self.Sz = jmat(S, 'z')

        # Nuclear operators
        self.Ix = jmat(I, 'x')
        self.Iy = jmat(I, 'y')
        self.Iz = jmat(I, 'z')

        self.Id_S = qeye(int(2*S + 1))
        self.Id_I = qeye(int(2*I + 1))



meas_Aperp = 51.
meas_Aperp = 48.

h    = 6.6260693e-34       # Plank constant
mu_0 = 12.566370614e-7     # Vacuum permeability
gamma_Er = np.array([117.3, 117.3, 17.45]) * 1e9 * h  # hyperfine coupling constants in Hz/T * h
mu_B = 9.27401007831e-24   # Bohr magneton in J/T
simu_A = np.array([[-441.66244757, -0.05970534,-6.00098845],[ -0.05970534, -441.66856909, 5.70123131],[ -6.00098845,5.70123131 ,131.30594568]])



def wrap_to_centered_cell(r, L):
    r = np.asarray(r, float)
    L = np.asarray(L, float)
    return (r + 0.5 * L) % L - 0.5 * L

def replicate_images(atoms, L, reps=(-1, 0, 1)):
    L = np.asarray(L, float)
    shifts = np.array([[i, j, k] for i in reps for j in reps for k in reps], float)
    out = []
    for s in shifts:
        shifted = np.empty(len(atoms), dtype=atoms.dtype)
        shifted["el"] = atoms["el"]
        shifted["xyz"] = atoms["xyz"] + s * L
        out.append(shifted)
    return np.concatenate(out)

def keep_near_cell(atoms, L, margin=1e-6):
    L = np.asarray(L, float)
    xyz = atoms["xyz"]
    lo = -0.5 * L - margin
    hi =  0.5 * L + margin
    m = np.all((xyz >= lo) & (xyz <= hi), axis=1)
    return atoms[m]

def plot_unit_cell(
    Crystal_atoms,
    L=np.array([5.334534, 5.334534, 11.50991]),
    atoms_to_plot=None,
    show_only=("Ca", "W", "Center_C"),
    ball_size=220,
    elev=18,
    azim=35,
    draw_cell=True,
    highlight_center=True,
    center_label="Center_C",
    center_size=900,
    depth_shading=True,
    depth_strength=0.35,
    cell_mode="centered",
    highlight_atom_index=None,
    highlight_color="green",
    highlight_size=650,
    figsize=(5.0, 4.0),     # <-- NEW: smaller default
    ax=None,                # <-- NEW: allow plotting into an existing axis
    show_legend=True,       # optional
):
    """
    Returns fig, ax, unitcell_atoms.
    If ax is provided, draws into that axis and returns its parent fig.
    Does NOT call plt.show().
    """

    if atoms_to_plot is None:
        atoms_to_plot = len(Crystal_atoms)
    atoms = Crystal_atoms[:int(atoms_to_plot)]

    if show_only is not None:
        atoms = atoms[np.isin(atoms["el"], np.array(show_only, dtype=str))]

    L = np.asarray(L, float)
    hx, hy, hz = 0.5 * L

    if cell_mode == "centered":
        cell_shift = np.array([0.0, 0.0, 0.0])
    elif cell_mode == "ca_corners":
        cell_shift = np.array([hx, hy, 0.0])
    else:
        raise ValueError("cell_mode must be 'centered' or 'ca_corners'")

    shifted = np.empty(len(atoms), dtype=atoms.dtype)
    shifted["el"] = atoms["el"]
    shifted["xyz"] = atoms["xyz"] - cell_shift

    wrapped = np.empty(len(shifted), dtype=shifted.dtype)
    wrapped["el"] = shifted["el"]
    wrapped["xyz"] = wrap_to_centered_cell(shifted["xyz"], L)

    images = replicate_images(wrapped, L, reps=(-1, 0, 1))
    unitcell_atoms = keep_near_cell(images, L, margin=1e-6)

    els = unitcell_atoms["el"]
    xyz = unitcell_atoms["xyz"].astype(float)

    base_colors = {
        "Ca": np.array([0.121, 0.466, 0.705, 1.0]),
        "W":  np.array([1.000, 0.498, 0.054, 1.0]),
        center_label: np.array([0.86, 0.08, 0.24, 1.0])
    }
    colors = np.array([base_colors.get(el, np.array([0.5, 0.5, 0.5, 1.0])) for el in els], float)

    if depth_shading and len(xyz) > 0:
        elr = np.deg2rad(elev)
        azr = np.deg2rad(azim)
        v = np.array([np.cos(elr)*np.cos(azr), np.cos(elr)*np.sin(azr), np.sin(elr)])
        depth = xyz @ v
        d0 = (depth - depth.min()) / (depth.max() - depth.min() + 1e-12)
        bright = (1.0 - depth_strength) + depth_strength * d0
        colors[:, :3] = colors[:, :3] * bright[:, None]

    # --- create fig/ax only if not supplied
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    ax.scatter(
        xyz[:, 0], xyz[:, 1], xyz[:, 2],
        s=ball_size,
        c=colors,
        depthshade=True,
        edgecolors="k",
        linewidths=0.35,
        alpha=1.0
    )

    if draw_cell:
        corners = np.array([
            [-hx, -hy, -hz], [ hx, -hy, -hz], [ hx,  hy, -hz], [-hx,  hy, -hz],
            [-hx, -hy,  hz], [ hx, -hy,  hz], [ hx,  hy,  hz], [-hx,  hy,  hz],
        ], float)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        for i, j in edges:
            ax.plot([corners[i, 0], corners[j, 0]],
                    [corners[i, 1], corners[j, 1]],
                    [corners[i, 2], corners[j, 2]],
                    linewidth=1.2, color="black", alpha=0.75)

    if highlight_center:
        ax.scatter([0.0], [0.0], [0.0],
                   s=center_size,
                   c=[base_colors.get(center_label, np.array([0.86, 0.08, 0.24, 1.0]))],
                   marker="*",
                   depthshade=False,
                   edgecolors="k",
                   linewidths=1.0,
                   alpha=1.0,
                   zorder=10)

    if highlight_atom_index is not None:
        idx0 = int(highlight_atom_index)
        if idx0 < 0 or idx0 >= len(Crystal_atoms):
            raise IndexError("highlight_atom_index is out of range for Crystal_atoms")

        r0 = np.array(Crystal_atoms["xyz"][idx0], dtype=float) - cell_shift
        r0w = wrap_to_centered_cell(r0, L)

        ax.scatter([r0w[0]], [r0w[1]], [r0w[2]],
                   s=highlight_size,
                   c=[highlight_color],
                   marker="o",
                   depthshade=False,
                   edgecolors="k",
                   linewidths=1.2,
                   alpha=1.0,
                   zorder=11)

    ax.set_xlabel("a (x) [Å]")
    ax.set_ylabel("b (y) [Å]")
    ax.set_zlabel("c (z) [Å]")
    ax.view_init(elev=elev, azim=azim)

    ax.set_xlim(-hx * 1.05, hx * 1.05)
    ax.set_ylim(-hy * 1.05, hy * 1.05)
    ax.set_zlim(-hz * 1.05, hz * 1.05)
    try:
        ax.set_box_aspect((L[0], L[1], L[2]))
    except Exception:
        pass

    if show_legend:
        unique_els = np.unique(els)
        handles, labels = [], []
        for el in unique_els:
            mk, ms = ("*", 10) if el == center_label else ("o", 8)
            fc = base_colors.get(el, np.array([0.5, 0.5, 0.5, 1.0]))
            handles.append(plt.Line2D([0],[0], marker=mk, linestyle="",
                                      markerfacecolor=fc[:3], markeredgecolor="k",
                                      markersize=ms))
            labels.append(el)
        if highlight_atom_index is not None:
            handles.append(plt.Line2D([0],[0], marker="o", linestyle="",
                                      markerfacecolor=highlight_color, markeredgecolor="k",
                                      markersize=8))
            labels.append(f"highlight {highlight_atom_index}")
        ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    return fig, ax, unitcell_atoms


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
    
def hyperfine_hamiltonian(sytem:SpinSystem, A) -> Qobj:
    h = 0 # Hyperfine interaction 
    for i, s_op in enumerate([sytem.Sx, sytem.Sy]):
        for j, i_op in enumerate([sytem.Ix, sytem.Iy, sytem.Iz]):
            h += simu_A[i, j] * tensor(s_op, i_op)
    return A * tensor(sytem.Sz, sytem.Iz) + meas_Aperp * tensor(sytem.Sz, sytem.Ix) + h

def quadrupole_hamiltonian_param(sytem:SpinSystem, D, E, Q, delta) -> Qobj:
    q_tensor = get_q_tensor(D, E, Q, delta)
    h = 0
    for i, i1 in enumerate([sytem.Ix, sytem.Iy, sytem.Iz]):
        for j, i2 in enumerate([sytem.Ix, sytem.Iy, sytem.Iz]):
            h += q_tensor[i, j] * tensor(sytem.Id_S, i1 * i2)
    return h

def zeeman_hamiltonian(sytem:SpinSystem, Bz) -> Qobj:
    return -Bz * (
        mu_Er * tensor(sytem.Sz, sytem.Id_I) +
        mu_Ca * tensor(sytem.Id_S, sytem.Iz)
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

def full_quadrupole_hamiltonian_param(sytem:SpinSystem, D, S1, S2, delta, theta) -> Qobj:
    q_tensor = get_full_q_tensor(D, S1, S2, delta, theta)
    h = 0
    for i, i1 in enumerate([sytem.Ix, sytem.Iy, sytem.Iz]):
        for j, i2 in enumerate([sytem.Ix, sytem.Iy, sytem.Iz]):
            h += q_tensor[i, j] * tensor(sytem.Id_S, i1*i2)
    return h

# Define the Hamiltonian
def Full_hamiltonian(x: np.ndarray, sytem:SpinSystem) -> Qobj: 
    Bz, A, D, S1, S2, delta, alpha, Dz  = x
    return zeeman_hamiltonian(sytem,Bz) +\
        hyperfine_hamiltonian(sytem,A) +\
        full_quadrupole_hamiltonian_param(sytem,D, S1, S2, delta, alpha) +\
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
    ndim = flat_samples.shape[0]
    high_low = np.zeros((ndim, 3))

    for i in range(ndim):
        p16, p50, p84 = np.percentile(flat_samples[i, :], [16, 50, 84])
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

def hexadecapole_hamiltonian(system:SpinSystem,Hx) -> Qobj:
    return Hx * tensor(system.Sz, system.Iz*system.Iz*system.Iz*system.Iz)

def sdq_hamiltonian_param(system:SpinSystem,Dz) -> Qobj:    
    q_tensor = get_full_q_tensor(Dz, 0,0,0,0)
    h = 0
    for i, i1 in enumerate([system.Ix, system.Iy, system.Iz]):
        for j, i2 in enumerate([system.Ix, system.Iy, system.Iz]):
            h += q_tensor[i, j] * tensor(system.Sz, i1*i2)
    return h

def hamiltonian(x: np.ndarray,system:SpinSystem) -> Qobj: 
    Bz, D, E, Q2, Q3  = x
    
    return zeeman_hamiltonian(system,Bz) + quadrupole_hamiltonian_param(system,D, E, Q2, Q3) 

def hamiltonian_Heca(x: np.ndarray,system:SpinSystem) -> Qobj: 

    Bz, D, E, Q2, Q3, Hx  = x
    
    return zeeman_hamiltonian(system,Bz) + quadrupole_hamiltonian_param(system,D, E, Q2, Q3) + hexadecapole_hamiltonian(system,Hx)

def get_hyperfine_tensor(g_electron, mu_I, xyz):
    """
    Calculates the hyperfine tensor and hamiltonian due to the dipole-dipole interaction between two spins
    The expression for the hyperfine tensor used is detailed in Le Dantec's Thesis (2022), p36
  
    """
    g_a, g_b, g_c = g_electron
    x, y, z = xyz
    r = np.linalg.norm(xyz)
    
    Tdd = np.zeros((3, 3)) # dipole-dipole tensor
    prefactor = r**-5 * mu_B * mu_I * mu_0/(4*np.pi)

    # diagonal
    Tdd[0,0] = g_a * (r**2 - 3*x**2)
    Tdd[1,1] = g_b * (r**2 - 3*y**2)
    Tdd[2,2] = g_c * (r**2 - 3*z**2)
    
    # xy
    Tdd[0,1] = g_a * (-3)*x*y
    Tdd[1,0] = g_b * (-3)*y*x
    
    # xz
    Tdd[0,2] = g_a * (-3)*x*z
    Tdd[2,0] = g_c * (-3)*z*x
    
    # yz
    Tdd[1,2] = g_b * (-3)*y*z
    Tdd[2,1] = g_c * (-3)*z*y
    
    Tdd *= prefactor * 1e-6 / h # [MHz]

    return -Tdd # be careful with the sign convention (see Arthur Schweiger and Gunnar Jeschke, "Principles of Pulse Electron Paramagnetic Resonance")

def _electron_axis(B_field, g_elec):
    
    B = np.asarray(B_field, float)
    if np.linalg.norm(B) == 0:
        raise ValueError("B_field must be nonzero.")
    G = np.asarray(g_elec, float)
    beff = (G.T @ B) if G.shape == (3,3) else (G * B)  # electron axis ∝ g^T · B
    return beff / np.linalg.norm(beff)

def Get_Rotated_B_field(B_field,ay = -0.57 * np.pi / 180, ax = -0.7 * np.pi / 180):

    def rotmat_xz(theta):
        return np.array([[np.cos(theta),  0, -np.sin(theta)],
                        [            0,  1,             0],
                        [np.sin(theta), 0, np.cos(theta)]])

    def rotmat_yz(theta):
        theta = -theta
        return np.array([[1,             0,              0],
                        [0, np.cos(theta), -np.sin(theta)],
                        [0, np.sin(theta),  np.cos(theta)]])

    return rotmat_yz(ay) @ rotmat_xz(ax) @ B_field


def get_rotations(mu1, mu2):
    """
    Returns rotation matrices for electron and nuclear spins.

    Parameters
    ----------
    mu1 : array_like
        Electron magnetic moment vector (arbitrary units).
    mu2 : array_like
        Nuclear magnetic moment vector (arbitrary units).

    Returns
    -------
    R_left : ndarray
        3x3 rotation matrix for electron spin. Rows are basis vectors [x, y, z].
    R_right : ndarray
        3x3 rotation matrix for nuclear spin. Rows are basis vectors [x', y', z'].
    """

    # Electron spin basis
    z = mu1 / np.linalg.norm(mu1)                   # z || mu1
    x = mu2 - np.dot(z, mu2) * z                   # component of mu2 perpendicular to z
    x /= np.linalg.norm(x)                          # normalize
    y = np.cross(z, x)                              # completes right-handed basis

    # Nuclear spin basis
    zp = mu2 / np.linalg.norm(mu2)                 # z' || mu2
    xp = mu1 - np.dot(zp, mu1) * zp               # component of mu1 perpendicular to zp
    xp /= np.linalg.norm(xp)                        # normalize
    yp = np.cross(zp, xp)                           # completes right-handed basis

    # Rotation matrices (rows = basis vectors)
    R_left = np.vstack([x, y, z])
    R_right = np.vstack([xp, yp, zp])
    
    # z_electron = mu1 / np.linalg.norm(mu1)  
    
    # y_electron = np.cross(z_electron , (0,0,1))
    # y_electron = y_electron/ np.linalg.norm(y_electron)  
    
    # x_electron =  np.cross(z_electron,y_electron)
    # x_electron = x_electron/ np.linalg.norm(x_electron)
  

    
    # print(x_electron)
    # print(y_electron)
    # print(z_electron)

    # z_nuclear = mu2 / np.linalg.norm(mu2)  
    # y_nuclear =  np.cross(z_nuclear , (0,0,1))
    # y_nuclear = y_nuclear/ np.linalg.norm(y_nuclear)  
    # x_nuclear =  np.cross(z_nuclear, y_nuclear)
    # x_nuclear = x_nuclear/ np.linalg.norm(x_nuclear)  

    
    # # Rotation matrices (rows = basis vectors)
    # R_left = np.vstack([x_electron, y_electron, z_electron])
    # R_right = np.vstack([x_nuclear, y_nuclear, z_nuclear])
        
    return R_left, R_right

# def Plot_hyperFine_for_site(thetas,b0,r,atome:str = None,phi_0 = 0.146 /360*2*np.pi, psi_0 = 0.368 /360*2*np.pi,site_index = 9,Aperp_to_plot=None,Apara_to_plot=None,Crystal_atoms = None):
def Plot_hyperFine_for_site(thetas,b0,r,atome:str = None,phi_0 = 0. /360*2*np.pi, psi_0 = 0. /360*2*np.pi,site_index = 9,Aperp_to_plot=None,Apara_to_plot=None,Crystal_atoms = None):
    # fig = visuals[0]
    print(phi_0, psi_0)
    A_paras  = []
    A_perps = []

    for theta in thetas :

        A_par, A_per,A_p = get_HyperFine(r,b0,theta,atome=atome,phi_0 = 0.146 /360*2*np.pi, psi_0 = 0.368 /360*2*np.pi,n_e= None,n_n=None)

        # print(f"theta: {theta*180/np.pi:.3f} deg, A_par: {A_par*1e3:.3f} kHz, A_perp: {A_per*1e3:.3f} kHz")
        A_paras.append(A_par)
        A_perps.append(A_per)

    if Crystal_atoms is not None:
        fig = plt.figure(figsize=(10, 4))

        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        plot_unit_cell(Crystal_atoms, ax=ax1, show_legend=True, cell_mode="centered",highlight_atom_index=site_index)

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(thetas,A_perps, label = 'Aperp')

        ax2.plot(thetas,A_paras, label = 'Apara')
        ax2.set_xlabel('theta [rad]')
        ax2.set_ylabel('A [MHz]')
        ax2.legend()
        
    else :

        ax2 = plt.subplot(111)
        ax2.plot(thetas,A_paras, label = 'Apara')
        ax2.plot(thetas,A_perps, label = 'Aperp')
        ax2.set_xlabel('theta [rad]')
        ax2.set_ylabel('A [MHz]')
        ax2.legend()
        if Apara_to_plot is not None:
            ax2.scatter(Apara_to_plot[:,0],Apara_to_plot[:,1],label = 'Apara from measurement')
        if Aperp_to_plot is not None:
            ax2.scatter(Aperp_to_plot[:,0],Aperp_to_plot[:,1],label = 'Aperp from measurement')

        plt.tight_layout()
        plt.show()



def get_HyperFine(r,b0,theta,atome:str = None,phi_0 = 0.146 /360*2*np.pi, psi_0 = 0.368 /360*2*np.pi,n_e= None,n_n=None):

    g_Er = gamma_Er / mu_B

    bx = b0 * (np.sin(theta) * np.sin(phi_0) - np.cos(theta) * np.sin(psi_0) * np.cos(phi_0))
    by = b0 * (np.sin(theta) * np.cos(phi_0) + np.cos(theta) * np.sin(psi_0) * np.sin(phi_0))
    bz = b0 * np.cos(theta) * np.cos(psi_0)

    rotated_b_field = Get_Rotated_B_field((bx, by, bz))

    if n_e is None and n_n is None:
        n_e = _electron_axis(rotated_b_field, g_Er) # electron axis || g^T B
        n_n = np.asarray(rotated_b_field, float)

    mu = mu_Ca if atome == 'Ca' else mu_Nb if atome == 'Nb' else mu_W
    Tdd = get_hyperfine_tensor(g_Er, mu, r)  # hyperfine tensor in crystal frame
    R_left, R_right = get_rotations(n_e, n_n)
    A_p = R_left @ Tdd @ R_right.T   # transpose on the nuclear rotationA_par = R_left @ Tdd @ R_right
    A_par = A_p[2,2]
    A_per = np.sqrt(A_p[2,0]**2+A_p[0,2]**2)

    return A_par, A_per, A_p

