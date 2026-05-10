import numpy as np
import matplotlib.pyplot as plt


h    = 6.6260693e-34       # Plank constant
mu_0 = 12.566370614e-7     # Vacuum permeability
gamma_Er = np.array([117.3, 117.3, 17.45]) * 1e9 * h  # hyperfine coupling constants in Hz/T * h
mu_B = 9.27401007831e-24   # Bohr magneton in J/T
simu_A = np.array([[-441.66244757, -0.05970534,-6.00098845],[ -0.05970534, -441.66856909, 5.70123131],[ -6.00098845,5.70123131 ,131.30594568]])



# def Plot_hyperFine_for_site(thetas,b0,r,atome:str = None,phi_0 = 0.146 /360*2*np.pi, psi_0 = 0.368 /360*2*np.pi,site_index = 9,Aperp_to_plot=None,Apara_to_plot=None,Crystal_atoms = None):
def Plot_hyperFine_for_site(thetas,b0,r,atome:str = None,phi_0 = 0. /360*2*np.pi, psi_0 = 0. /360*2*np.pi,site_index = 9,Aperp_to_plot=None,Apara_to_plot=None,Crystal_atoms = None):
    # fig = visuals[0]
    print(phi_0, psi_0)
    A_paras  = []
    A_perps = []

    for theta in thetas :

        A_par, A_per,A_p = get_HyperFine(r,theta,atome=atome)

        # print(f"theta: {theta*180/np.pi:.3f} deg, A_par: {A_par*1e3:.3f} kHz, A_perp: {A_per*1e3:.3f} kHz")
        A_paras.append(A_par)
        A_perps.append(A_per)
    
    A_paras = np.array(A_paras)
    A_perps = np.array(A_perps)

    if Crystal_atoms is not None:
        fig = plt.figure(figsize=(10, 4))

        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        plot_unit_cell(Crystal_atoms, ax=ax1, show_legend=True, cell_mode="centered",highlight_atom_index=site_index)

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



def get_HyperFine(r,theta,atome:str = None):

    g_Er = gamma_Er / mu_B

    mu = mu_Ca if atome == 'Ca' else mu_Nb if atome == 'Nb' else mu_W

    B_direction = np.array([np.sin(theta),0,np.cos(theta)])
    R = Get_nuc_to_elec_rotation(g_Er, B_direction)
    Hyperfine_tensor = get_hyperfine_tensor(g_Er, mu, r)  # hyperfine tensor in crystal frame
    Hyperfine_tensor_rot = R @ Hyperfine_tensor @ R.T

    A_par = Hyperfine_tensor_rot[2, 2]
    A_per = Hyperfine_tensor_rot[2, 0]

    return A_par, A_per, Hyperfine_tensor_rot


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