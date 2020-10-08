from leapfrog import leapfrog
import scipy
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.ticker import NullFormatter
import astropy.units as u
import astropy.constants as c
import ipyvolume as ipv

# Set initial parameters
M_A = 1e6*u.Msun
M_B = 1e6*u.Msun
r_A = 15*u.kpc
r_B = 15*u.kpc
theta_A = 45
theta_B = 0
phi_A = 30
phi_B = 0
npoints=1000
N = 31
dt = 0.5
# Get initial positions and velocities
xyz_A, v_xyz_A = initial_plummer_positions(npoints, M=M_A, seed=798632, theta=theta_A, phi=phi_A, radius=r_A)
xyz_B, v_xyz_B = initial_plummer_positions(npoints, M=M_B, seed=798632, theta=theta_B, phi=phi_B, radius=r_B)
# Shift the galaxies away from each other
pos_A = [-1.2*r_A.value, 0, 0]
pos_B = [1.2*r_B.value, 0, 0]
A_xyz = [np.append(xyz_A[i], 0) + pos_A[i] for i in range(3)]*u.kpc
B_xyz = [np.append(xyz_B[i], 0) + pos_B[i] for i in range(3)]*u.kpc
# Plot initial conditions
plt.style.use('dark_background')
fig,axes = plt.subplots(ncols = 3, figsize=(15,5))
axes[0].scatter(A_xyz[0], A_xyz[1], s=20, marker="*", c="skyblue")
axes[0].scatter(B_xyz[0], B_xyz[1], s=20, marker="*", c="white")
axes[0].set_xlabel('x [kpc]')
axes[0].set_ylabel('y [kpc]')
axes[1].scatter(A_xyz[0], A_xyz[2], s=20, marker="*", c="skyblue")
axes[1].scatter(B_xyz[0], B_xyz[2], s=20, marker="*", c="white")
axes[1].set_xlabel('x [kpc]')
axes[1].set_ylabel('z [kpc]')
axes[2].scatter(A_xyz[1], A_xyz[2], s=20, marker="*", c="skyblue")
axes[2].scatter(B_xyz[1], B_xyz[2], s=20, marker="*", c="white")
axes[2].set_xlabel('y [kpc]')
axes[2].set_ylabel('z [kpc]')
for i in [0, 1, 2]:
    axes[i].set_ylim((-35, 35))
    axes[i].set_xlim((-35, 35))
fig.tight_layout()
# Run leapfrog of barnes hut
pos_t, vel_t = leapfrog(A_xyz, B_xyz, v_xyz_A, v_xyz_B, M_A=1e6*u.Msun, M_B=6e6*u.Msun, npoints=npoints, N=N, dt=dt)
# Plot spatial evolution of merger
plt.style.use('dark_background')
fig,axes = plt.subplots(ncols = 5, nrows=2, figsize=(20, 7))
axes[0][0].scatter(pos_t[0].T[0][:npoints], pos_t[0].T[1][:npoints], s=20, marker="*", c="white")
axes[0][0].scatter(pos_t[0].T[0][npoints:], pos_t[0].T[1][npoints:], s=20, marker="*", c="skyblue")
axes[0][0].set_ylabel('y [kpc]')
axes[0][0].ticklabel_format(style='sci',scilimits=(-3,3),axis='both')
axes[1][0].scatter(pos_t[0].T[0][:npoints], pos_t[0].T[2][:npoints], s=20, marker="*", c="white")
axes[1][0].scatter(pos_t[0].T[0][npoints:], pos_t[0].T[2][npoints:], s=20, marker="*", c="skyblue")
axes[1][0].set_ylabel('z [kpc]')
axes[1][0].set_xlabel('x [kpc]')
t_range = np.arange(1, (N-1)/dt, ((N-1)/dt)/5)
for i, t in zip(range(1, 5), t_range):
    t = int(t)
    for k in range(2):
        axes[k][i].scatter(pos_t[t].T[0][1001:], pos_t[t].T[k+1][1001:], s=20, marker="*", c="skyblue")
        axes[k][i].scatter(pos_t[t].T[0][:1001], pos_t[t].T[k+1][:1001], s=20, marker="*", c="white")
        axes[1][i].set_xlabel('x [kpc]')
fig.tight_layout()