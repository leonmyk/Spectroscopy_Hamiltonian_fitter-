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
def Plot_hyperFine_for_site(thetas, b0, r, mu, Aperp_to_plot=None, Apara_to_plot=None, Crystal_atoms=None,labels = None,basis_vecs = [],save = False,animate = False,lims = [1.1,1.1,1.1],measured_Angle = None,measured_A =None,measured_B =None):
    A_paras = []
    A_perps = []

    for theta in thetas:
        A_par, A_per, A_p = get_HyperFine(r, theta, mu)
        A_paras.append(A_par)
        A_perps.append(A_per)

    A_paras = np.array(A_paras)
    A_perps = np.array(A_perps)

    colors = ['blue','green','red']

    if Crystal_atoms is not None:
        fig = plt.figure(figsize=(15, 7))

        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        plot_crystal(ax=ax1, Crystal_atoms=Crystal_atoms, r=r*1e10,lims=lims)

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(thetas*180/np.pi, A_paras*1e3, label='A')
        ax2.plot(thetas*180/np.pi, A_perps*1e3, label='B')
        ax2.scatter(measured_Angle,measured_A,color = 'blue', label = 'Measured A')
        ax2.scatter(measured_Angle,measured_B, color = 'orange', label = 'Measured B')
        ax2.set_xlabel(r'$\theta$  [deg from C axis]')
        ax2.set_ylabel('A [KHz]')
        ax2.legend()

        for i,v in enumerate(basis_vecs):
            ax1.quiver(
            *(r*1e10),
            v[0]*2, v[1]*2, v[2]*2,
            color=colors[i],
            linewidth=2,
            arrow_length_ratio=0.15
            )
            v = v+(r*1e10)

            # label at tip
            ax1.text(
                v[0], v[1], v[2],
                labels[i],
                color=colors[i]
            )

        def update(frame):
                    ax1.view_init(elev=20, azim=frame)
                    ax1.set_box_aspect((1, 1, 2))
                    # ax1.legend(loc='upper left', bbox_to_anchor=(-0.35, 1), bbox_transform=ax1.transAxes)
                    # ax1.legend()
                    # ax1.set_xlim(-a/2 * 1.1, a/2 * 1.1)
                    # ax1.set_ylim(-a/2 * 1.1, a/2 * 1.1)
                    # ax1.set_zlim(-6.3, 7)
        ani = None
        if animate:
        
            from matplotlib.animation import FuncAnimation
            ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 1), interval=100, blit=False, repeat=True)
            if save:
                ani.save('crystal_rotation.gif', writer='pillow', fps=30)

        plt.show()
        return ani
    
    else:
        ax2 = plt.subplot(111)
        ax2.plot(thetas*180/np.pi, A_paras*1e3, label='A')
        ax2.plot(thetas*180/np.pi, A_perps*1e3, label='B')
        ax2.set_xlabel('theta [deg]')
        ax2.set_ylabel('A [KHz]')
        ax2.legend()
        if Apara_to_plot is not None:
            ax2.scatter(Apara_to_plot[:,0], Apara_to_plot[:,1], label='Apara from measurement')
        if Aperp_to_plot is not None:
            ax2.scatter(Aperp_to_plot[:,0], Aperp_to_plot[:,1], label='Aperp from measurement')
        plt.show()




def get_HyperFine(r,theta,mu):
    #returns the hyperfine tensor in the rotated frame

    g_Er = gamma_Er / mu_B


    B_direction = np.array([0,np.sin(theta),np.cos(theta)])
    R = Get_nuc_to_elec_rotation(g_Er, B_direction)
    Hyperfine_tensor = get_hyperfine_tensor(g_Er, mu, r)  # hyperfine tensor in crystal frame
    Hyperfine_tensor_rot = R @ Hyperfine_tensor @ R.T

    A_par = Hyperfine_tensor_rot[2, 2]
    A_per = Hyperfine_tensor_rot[2, 0] # This is because the field B is [sin,0,cos] if it was [0,sin,cos] it would be Hyperfine_tensor_rot[2, 1]

    return A_par, A_per, Hyperfine_tensor_rot


def plot_crystal(ax, Crystal_atoms, r,lims = None):

    def plot_vec(ax, p1, p2, **kwargs):
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            **kwargs
        )

    c = 5.754955 * 2
    a = 5.3

    Ca_positions = Crystal_atoms[Crystal_atoms["el"] == "Ca"]["xyz"]
    W_positions  = Crystal_atoms[Crystal_atoms["el"] == "W"]["xyz"]
    Erbium_position = Crystal_atoms[Crystal_atoms["el"] == "Center_C"]["xyz"]

    fake_Ca = Ca_positions + np.array([0, 0, c])
    fake_W  = W_positions  + np.array([0, 0, c])

    Ca_all = np.vstack([Ca_positions, fake_Ca])
    W_all  = np.vstack([W_positions,  fake_W])

    Ca_bg = Ca_all[~np.all(np.abs(Ca_all - r) < 0.1, axis=1)]
    W_bg  = W_all[~np.all(np.abs(W_all  - r) < 0.1, axis=1)]

    def in_cell(pos):
        return (
            (pos[:, 2] > -c/2 * 1.05) & (pos[:, 2] < c/2 * 1.05) &
            (np.abs(pos[:, 0]) < a/2 * 1.05) &
            (np.abs(pos[:, 1]) < a/2 * 1.05)
        )

    def draw_frame():
        Ca_bg_in_cell = Ca_bg[in_cell(Ca_bg)]
        Ca_bg_in_cell = np.delete(Ca_bg_in_cell, 5, axis=0)
        ax.scatter(*Ca_bg_in_cell.T,  c='blue',   s=100, label='Ca')
        ax.scatter(*W_bg[in_cell(W_bg)].T,    c='orange', s=100, label='W')
        ax.scatter(r[0], r[1], r[2], s=25, color='red', alpha=1,
                   zorder=10, linewidth=10, label='selected')

        ax.set_xlabel('X (A°)')
        ax.set_ylabel('Y (A°)')
        ax.set_zlabel('Z (A°)')

        for i, j in [(8, 7), (7, 5), (5, 6), (6, 8)]:
            plot_vec(ax, Ca_all[i],   Ca_all[j],   color='blue')
            plot_vec(ax, Ca_all[i+9], Ca_all[j+9], color='blue')
            plot_vec(ax, Ca_all[i],   Ca_all[i+9], color='black')

        for i, j in [(2, 4), (4, 8), (8, 6), (6, 2)]:
            plot_vec(ax, W_all[i], W_all[j], color='orange')

        for i, j in [(17, 4), (15, 8), (16, 8), (17, 6), (17, 2), (14, 6), (14, 4), (15, 2)]:
            plot_vec(ax, Ca_all[i], W_all[j], color='grey', alpha=0.9)

        for i, j in [(8, 4), (6, 8), (7, 8), (8, 6), (7, 2), (5, 6), (5, 4), (6, 2)]:
            plot_vec(ax, Ca_all[i], W_all[j], color='grey', alpha=0.9)

        for ca_idx in [5, 6, 7, 8]:
            plot_vec(ax, W_all[12], Ca_all[ca_idx], color='dimgrey', alpha=0.7, linewidth=1)

        for ca_idx in [14, 15, 16, 17]:
            plot_vec(ax, W_all[25], Ca_all[ca_idx], color='dimgrey', alpha=0.7, linewidth=1)

        ax.scatter(Erbium_position[0, 0], Erbium_position[0, 1], Erbium_position[0, 2],
                   c='purple', label='Er', s=100)
        ax.set_box_aspect((1, 1, 2))
        ax.legend(loc='upper left', bbox_to_anchor=(-0.35, 1), bbox_transform=ax.transAxes)

        ax.set_xlim(-a/2 * lims[0], a/2 *  lims[0])
        ax.set_ylim(-a/2 *  lims[1], a/2 *  lims[1])
        ax.set_zlim(-c/2 *  lims[2], c/2 *  lims[2])

    draw_frame()
    