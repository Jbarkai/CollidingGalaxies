
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

def plot_init_vels(A_xyz, B_xyz, A_v_xyz, B_v_xyz, lim=35):
    fig,axes = plt.subplots(ncols=3, figsize=(15,5))
    axes[0].quiver(A_xyz[0], A_xyz[1], A_v_xyz[0], A_v_xyz[1], color="maroon", label="Galaxy A")
    axes[0].quiver(B_xyz[0], B_xyz[1], B_v_xyz[0], B_v_xyz[1], color="green", label="Galaxy B")
    # axes[0].scatter(A_xyz[0][-1], A_xyz[1][-1], color="black", marker="*", label="Black Hole A")
    # axes[0].scatter(B_xyz[0][-1], B_xyz[1][-1], color="black", label="Black Hole B")
    axes[0].set_xlabel('x [kpc]')
    axes[0].set_ylabel('y [kpc]')
    axes[1].quiver(A_xyz[0], A_xyz[2], A_v_xyz[0], A_v_xyz[2], color="maroon", label="Galaxy A")
    axes[1].quiver(B_xyz[0], B_xyz[2], B_v_xyz[0], B_v_xyz[2], color="green", label="Galaxy B")
    # axes[1].scatter(A_xyz[0][-1], A_xyz[2][-1], color="black", marker="*", label="Black Hole A")
    # axes[1].scatter(B_xyz[0][-1], B_xyz[2][-1], color="black", label="Black Hole B")
    axes[1].set_xlabel('x [kpc]')
    axes[1].set_ylabel('z [kpc]')
    axes[2].quiver(A_xyz[1], A_xyz[2], A_v_xyz[1], A_v_xyz[2], color="maroon", label="Galaxy A")
    axes[2].quiver(B_xyz[1], B_xyz[2], B_v_xyz[1], B_v_xyz[2], color="green", label="Galaxy B")
    # axes[2].scatter(A_xyz[1][-1], A_xyz[2][-1], color="black", marker="*", label="Black Hole A")
    # axes[2].scatter(B_xyz[1][-1], B_xyz[2][-1], color="black", label="Black Hole B")
    axes[2].set_xlabel('y [kpc]')
    axes[2].set_ylabel('z [kpc]')
    for i in [0, 1, 2]:
        axes[i].set_ylim((-lim, lim))
        axes[i].set_xlim((-lim, lim))
        axes[i].legend()
    fig.tight_layout()
    plt.show()

def plot_evol(pos_t, npoints, N, dt):
    # Plot evolution of merger
    fig,axes = plt.subplots(ncols = 6, nrows=2, figsize=(50, 20))
    axes[0][0].scatter(pos_t[0].T[0][:npoints], pos_t[0].T[1][:npoints], s=8,
                   color="maroon", label="t=0Gyr")
    axes[0][0].scatter(pos_t[0].T[0][npoints:], pos_t[0].T[1][npoints:], s=8,
                   color="green")
    axes[0][0].set_ylabel('y [kpc]', fontsize=30)
    axes[0][0].legend(fontsize=30)
    axes[0][0].set_xlabel('x [kpc]', fontsize=30)

    axes[1][0].scatter(pos_t[0].T[0][:npoints], pos_t[0].T[2][:npoints], s=8,
                   color="maroon", label="t=0Gyr")
    axes[1][0].scatter(pos_t[0].T[0][npoints:], pos_t[0].T[2][npoints:], s=8,
                   color="green")
    axes[1][0].set_ylabel('z [kpc]', fontsize=30)
    axes[1][0].legend(fontsize=30)
    axes[1][0].set_xlabel('x [kpc]', fontsize=30)

    t_range = np.arange(N/5, N+1, N/5)
    for i, t in zip(range(1, 6), t_range):
        t = int(t) - 1
        axes[0][i].scatter(pos_t[t].T[0][:npoints], pos_t[t].T[1][:npoints], s=8,
                       color="maroon", label="t=%s" %(t*dt))
        axes[0][i].scatter(pos_t[t].T[0][npoints:], pos_t[t].T[1][npoints:], s=8,
                       color="green")
        axes[0][i].set_xlabel('x [kpc]', fontsize=30)
        axes[0][i].legend(fontsize=30)

        axes[1][i].scatter(pos_t[t].T[0][:npoints], pos_t[t].T[2][:npoints], s=8,
                       color="maroon", label="t=%s" %(t*dt))
        axes[1][i].scatter(pos_t[t].T[0][npoints:], pos_t[t].T[2][npoints:], s=8,
                       color="green")
        axes[1][i].set_xlabel('x [kpc]', fontsize=30)
        axes[1][i].legend(fontsize=30)
    fig.tight_layout()
    plt.show()