import numpy as np
import matplotlib.pyplot as plt
from Hamiltonians import Get_nuc_to_elec_rotation
from Hamiltonians import get_hyperfine_tensor



h    = 6.6260693e-34       # Plank constant
mu_0 = 12.566370614e-7     # Vacuum permeability
gamma_Er = np.array([117.3, 117.3, 17.45]) * 1e9 * h  # hyperfine coupling constants in Hz/T * h
mu_B = 9.27401007831e-24   # Bohr magneton in J/T
simu_A = np.array([[-441.66244757, -0.05970534,-6.00098845],[ -0.05970534, -441.66856909, 5.70123131],[ -6.00098845,5.70123131 ,131.30594568]])



# def Plot_hyperFine_for_site(thetas,b0,r,atome:str = None,phi_0 = 0.146 /360*2*np.pi, psi_0 = 0.368 /360*2*np.pi,site_index = 9,Aperp_to_plot=None,Apara_to_plot=None,Crystal_atoms = None):
def Plot_hyperFine_for_site(thetas,b0,r,mu,Aperp_to_plot=None,Apara_to_plot=None,Crystal_atoms = None):
    # fig = visuals[0]
    A_paras  = []
    A_perps = []

    for theta in thetas :

        A_par, A_per,A_p = get_HyperFine(r,theta,mu)

        # print(f"theta: {theta*180/np.pi:.3f} deg, A_par: {A_par*1e3:.3f} kHz, A_perp: {A_per*1e3:.3f} kHz")
        A_paras.append(A_par)
        A_perps.append(A_per)
    
    A_paras = np.array(A_paras)
    A_perps = np.array(A_perps)

    if Crystal_atoms is not None:
        fig = plt.figure(figsize=(10, 4))

        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        plot_crystal(ax=ax1,Crystal_atoms=Crystal_atoms,r = r)

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(thetas*180/np.pi,A_perps*1e3, label = 'Aperp')

        ax2.plot(thetas*180/np.pi,A_paras*1e3, label = 'Apara')
        ax2.set_xlabel('theta [deg]')
        ax2.set_ylabel('A [KHz]')
        ax2.legend()
        
    else :

        ax2 = plt.subplot(111)
        ax2.plot(thetas*180/np.pi,A_paras*1e3, label = 'Apara')
        ax2.plot(thetas*180/np.pi,A_perps*1e3, label = 'Aperp')
        ax2.set_xlabel('theta [deg]')
        ax2.set_ylabel('A [KHz]')
        ax2.legend()
        if Apara_to_plot is not None:
            ax2.scatter(Apara_to_plot[:,0],Apara_to_plot[:,1],label = 'Apara from measurement')
        if Aperp_to_plot is not None:
            ax2.scatter(Aperp_to_plot[:,0],Aperp_to_plot[:,1],label = 'Aperp from measurement')

        plt.tight_layout()
        plt.show()



def get_HyperFine(r,theta,mu):

    g_Er = gamma_Er / mu_B


    B_direction = np.array([np.sin(theta),0,np.cos(theta)])
    R = Get_nuc_to_elec_rotation(g_Er, B_direction)
    Hyperfine_tensor = get_hyperfine_tensor(g_Er, mu, r)  # hyperfine tensor in crystal frame
    Hyperfine_tensor_rot = R @ Hyperfine_tensor @ R.T

    A_par = Hyperfine_tensor_rot[2, 2]
    A_per = Hyperfine_tensor_rot[2, 0] # This is because the field B is [sin,0,cos] if it was [0,sin,cos] it would be Hyperfine_tensor_rot[2, 1]

    return A_par, A_per, Hyperfine_tensor_rot



def plot_crystal(ax,Crystal_atoms,r):


    def plot_vec(ax,p1, p2, **kwargs):
        ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                [p1[2], p2[2]],
                **kwargs
        )

    c = 5.754955*2
    a = 5.3 

    r = np.array([0,0,c/2])

    Ca_positions = Crystal_atoms[Crystal_atoms["el"] == "Ca"]["xyz"]
    W_positions = Crystal_atoms[Crystal_atoms["el"] == "W"]["xyz"]
    Erbium_position = Crystal_atoms[Crystal_atoms["el"] == "Center_C"]["xyz"]
    fake_Ca = Ca_positions + np.array([0,0,c])
    fake_W = W_positions + np.array([0,0,c])

    filtered_Ca_positions = Ca_positions[np.where((Ca_positions[:,2] > -c/2*1.05) & (Ca_positions[:,2] < c/2*1.05) & (np.abs(Ca_positions[:,0]) < a/2*1.05) & (np.abs(Ca_positions[:,1]) < a/2*1.05))]
    filtered_W_positions = W_positions[np.where((W_positions[:,2] > -c/2*1.05) & (W_positions[:,2] < c/2*1.05) & (np.abs(W_positions[:,0]) < a/2*1.05) & (np.abs(W_positions[:,1]) < a/2*1.05))]
    filtered_fake_Ca = fake_Ca[np.where((fake_Ca[:,2] > -c/2*1.05) & (fake_Ca[:,2] < c/2*1.05) & (np.abs(fake_Ca[:,0]) < a/2*1.05) & (np.abs(fake_Ca[:,1]) < a/2*1.05))]
    filtered_fake_W = fake_W[np.where((fake_W[:,2] > -c/2*1.05) & (fake_W[:,2] < c/2*1.05) & (np.abs(fake_W[:,0]) < a/2*1.05) & (np.abs(fake_W[:,1]) < a/2*1.05))]

    ax.scatter(filtered_fake_Ca[:,0], filtered_fake_Ca[:,1], filtered_fake_Ca[:,2], c='blue',s=100)
    ax.scatter(filtered_fake_W[:,0], filtered_fake_W[:,1], filtered_fake_W[:,2], c='orange', s=100)
    ax.scatter(filtered_Ca_positions[:,0], filtered_Ca_positions[:,1], filtered_Ca_positions[:,2], c='blue', label=' Ca',s=100, )
    ax.scatter(filtered_W_positions[:,0], filtered_W_positions[:,1], filtered_W_positions[:,2], c='orange', label=' W',s=100,)
    ax.scatter(r[0], r[1], r[2], s=25, color='red', alpha=1, zorder=10,linewidth=10,label  = 'selected')


    ax.set_xlabel('X (nm)')
    ax.set_ylabel('Y (nm)')
    ax.set_zlabel('Z (nm)')
    ax.legend()


    for i in [(-1, -2), (-2, -4), (-4, -3), (-3, -1)]:
        plot_vec(ax, Ca_positions[i[0]], Ca_positions[i[1]], color = 'blue')
        plot_vec(ax, fake_Ca[i[0]], fake_Ca[i[1]], color = 'blue')
        plot_vec(ax, Ca_positions[i[0]],fake_Ca[i[0]], color = 'black')


    for i in [(2, 4), (4, 8), (8, 6), (6, 2)]:
        plot_vec(ax, W_positions[i[0]], W_positions[i[1]], color = 'orange')



    ax.scatter(Erbium_position[0,0], Erbium_position[0,1], Erbium_position[0,2], c='red', label='Er')
    ax.set_box_aspect((1,1,2))

    ax.set_xlim(-a/2*1.1, a/2*1.1)
    ax.set_ylim(-a/2*1.1, a/2*1.1)
    ax.set_zlim(-c/2*1.1, c/2*1.1)


