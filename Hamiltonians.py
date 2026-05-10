import numpy as np
from qutip import *



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
mu_Ca = -2.86899 #[kHz / mT]

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


def hyperfine_hamiltonian(system:SpinSystem, A, meas_Aperp) -> Qobj:
    h = 0 # Hyperfine interaction 
    for i, s_op in enumerate([system.Sx, system.Sy]):
        for j, i_op in enumerate([system.Ix, system.Iy, system.Iz]):
            h += simu_A[i, j] * tensor(s_op, i_op)
    return A * tensor(system.Sz, system.Iz) + meas_Aperp * tensor(system.Sz, system.Ix) + h

def hyperfine_hamiltonian_V2(system:SpinSystem,B0, A_para, meas_Aperp) -> Qobj:


    hyperfine_tensor = 0 # Hyperfine interaction 
    hyperfine_matrix = np.diag(meas_Aperp,meas_Aperp,A_para)
    for i in range(3):
        for j in range(i):
            hyperfine_matrix[i,j] = system.simu_A[i, j]
            hyperfine_matrix[j,i] = system.simu_A[j, i]
    
    R = Get_nuc_to_elec_rotation(system.g_Er, B0)
    rotated_hyperfine_matrix = R @ hyperfine_matrix @ R.conj().T


    for i, s_op in enumerate([system.Sx, system.Sy,system.Sz]):
        for j, i_op in enumerate([system.Ix, system.Iy, system.Iz]):
            hyperfine_tensor += rotated_hyperfine_matrix[i, j] * tensor(s_op, i_op)


    return hyperfine_tensor

def quadrupole_hamiltonian_param(system:SpinSystem, D, E, Q, delta) -> Qobj:
    q_tensor = get_q_tensor(D, E, Q, delta)
    h = 0
    for i, i1 in enumerate([system.Ix, system.Iy, system.Iz]):
        for j, i2 in enumerate([system.Ix, system.Iy, system.Iz]):
            h += q_tensor[i, j] * tensor(system.Id_S, i1 * i2)
    return h

def zeeman_hamiltonian(system:SpinSystem, Bz) -> Qobj:
    return -np.sign(system.mu)*Bz * (
        mu_Er * tensor(system.Sz, system.Id_I) +
        system.mu * tensor(system.Id_S, system.Iz)
    )

    
def get_q_tensor(D, E, Q, delta):
    c = E * np.cos(2 * delta)
    s = E * np.sin(2 * delta)
    q_tensor = np.array([
        [-D/2 + c,  s, Q],
        [s, -D/2 - c, 0],
        [Q,     0,    D]
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

def full_quadrupole_hamiltonian_param(system:SpinSystem, D, S1, S2, delta, theta) -> Qobj:
    q_tensor = get_full_q_tensor(D, S1, S2, delta, theta)
    h = 0
    for i, i1 in enumerate([system.Ix, system.Iy, system.Iz]):
        for j, i2 in enumerate([system.Ix, system.Iy, system.Iz]):
            h += q_tensor[i, j] * tensor(system.Id_S, i1*i2)
    print(q_tensor)
    return h

# Define the Hamiltonian
def Full_hamiltonian(x: np.ndarray, system:SpinSystem,meas_Aperp) -> Qobj: 
    Bz, A, D, S1, S2, delta, alpha, Dz , meas_Aperp = x
    return zeeman_hamiltonian(system,Bz) +\
        hyperfine_hamiltonian(system,A,meas_Aperp) +\
        full_quadrupole_hamiltonian_param(system,D, S1, S2, delta, alpha) +\
        sdq_hamiltonian_param(system,Dz) #+\
        #hexadecapole_hamiltonian(Hx)


def Full_hamiltonian_V2(x: np.ndarray, system:SpinSystem ,meas_Aperp,angle) -> Qobj: 
    B0, A_para, D, S1, S2, delta, alpha, Dz  = x


    B = np.array([np.sin(angle),0,np.cos(angle)])*B0

    return zeeman_hamiltonian_V2(system,B,angle) +\
        hyperfine_hamiltonian_V2(system,B,A_para,meas_Aperp) +\
        full_quadrupole_hamiltonian_param(system,D, S1, S2, delta, alpha) +\
        sdq_hamiltonian_param(system,Dz) #+\


def Get_nuc_to_elec_rotation(elec_g_tensor,B):

    ez = elec_g_tensor*B/np.linalg.norm(elec_g_tensor*B)
    ex = np.cross(np.array([0,1,0]), ez)/np.linalg.norm(np.cross(np.array([0,1,0]), ez))
    ey = np.cross(ez,ex)

    rotation_matrix = np.vstack((ex,ey,ez))

    return rotation_matrix


def zeeman_hamiltonian_V2(system: SpinSystem, B, angle) -> Qobj:

    R = Get_nuc_to_elec_rotation(system.g_Er, B) # rotation makes the basis align with the electron effective magnetic field 

    Beffz = np.linalg.norm(R @ (system.g_Er * B)) #effective field of the electron 

    B_rot = R @ B  # the field perceived by the nuclear spin which is different to the electron because of the anisotropic g tensor 

    elec_term = -Beffz * mu_B * tensor(system.Sz, system.Id_I)

    nuc_term = -system.mu * sum(
        Bi * tensor(system.Id_S, Ii)
        for Bi, Ii in zip(B_rot, [system.Ix, system.Iy, system.Iz])
    )

    return elec_term + nuc_term




class SpinSystem:

    def __init__(self,mu:str,simu_A, S=1/2, I=7/2):
        self.S = S
        self.I = I

        self.gamma_Er = np.array([117.3, 117.3, 17.45]) * 1e9 * h  # hyperfine coupling constants in Hz/T * h
        self.g_Er = gamma_Er / mu_B

        self.simu_A = simu_A

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

        self.S_ops = np.array([
            tensor(self.Sx, self.Id_I),
            tensor(self.Sy, self.Id_I),
            tensor(self.Sz, self.Id_I)
        ])

        self.I_ops = np.array([
            tensor(self.Id_S, self.Ix),
            tensor(self.Id_S, self.Iy),
            tensor(self.Id_S, self.Iz)
        ])
        
        if mu == "Nb":
            self.mu = mu_Nb
        elif mu == "Ca":
            self.mu = mu_Ca
            
        


