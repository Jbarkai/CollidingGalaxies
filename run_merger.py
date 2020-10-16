from leapfrog import leapfrog
import scipy
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.ticker import NullFormatter
import astropy.units as u
import astropy.constants as c
from plotters import plot_init, plot_evol, plot_init_vels, plot_evol_vels
from initial_conditions import initial_plummer_positions

# Set initial parameters
r_A = 15*u.kpc
r_B = 15*u.kpc
a_A = 8*u.kpc
a_B = 8*u.kpc
A_vel = 0
B_vel = -0
npoints=100
N = 50
dt = 1*u.Gyr
softening = 0.98*npoints**(-0.26)
# SITUATION 1 Ma:Mb=1:1
M_A = 1e6*u.Msun
M_B = 1e6*u.Msun
# Get initial positions and velocities of the two systems
xyz_A, v_xyz_A, rs_A, vr_A = initial_plummer_positions(
    npoints, M=M_A, seed=798632, radius=r_A, a=a_A,
    x_pos=0, z_pos=-0, x_vel=A_vel
)
xyz_B, v_xyz_B, rs_B, vr_B = initial_plummer_positions(
    npoints, M=M_B, seed=987344, radius=r_B, a=a_B,
    x_pos=1*r_B.value, x_vel=B_vel
)
# Check radial velocity
plt.scatter(rs_A, (vr_A*u.kpc/u.s).to(u.km/u.s).value, s=8, c='maroon')
plt.xlabel("Radius r [kpc]", fontsize=14)
plt.ylabel(r"Radial Velocity v$_r$ [km/s]", fontsize=14)
plt.show()
# Plot initial conditions
plot_init_vels(xyz_A, xyz_B, v_xyz_A, v_xyz_B, lim=35)

# Combine positions, velocities and masses of two systems
ab_pos = np.concatenate((xyz_A.T, xyz_B.T))[:2*npoints]
ab_vel = np.concatenate((v_xyz_A.T, v_xyz_B.T))[:2*npoints]
# Assume the particles are of equal mass
masses_A = np.repeat((M_A.value/npoints), len(ab_pos)/2)
masses_B = np.repeat((M_B.value/npoints), len(ab_pos)/2)
# masses = np.concatenate(masses_A, masses_B)
masses = np.array(list(masses_A) + list(masses_B))
print("Time ellapsed = ", N*dt)
# Test integration on one galaxy
# Run leapfrog of barnes hut
pos_t_test, vel_t_test = leapfrog(xyz_A.T, v_xyz_A.T, masses_A, softening=softening, N=N, dt=dt)
# Plot evolution of merger
fig,axes = plt.subplots(ncols = 6, figsize=(20, 4))
axes[0].quiver(pos_t_test[0].T[0], pos_t_test[0].T[1],
                  vel_t_test[0].T[0], vel_t_test[0].T[1],
                  color="maroon", label="t=0Gyr")
axes[0].set_ylabel('y [kpc]')
axes[0].set_xlabel('x [kpc]')
t_range = np.arange(N/5, N+1, N/5)
for i, t in zip(range(1, 6), t_range):
    t = int(t) - 1
    axes[i].quiver(pos_t_test[t].T[0], pos_t_test[t].T[1],
              vel_t_test[t].T[0], vel_t_test[t].T[1],
                   color="maroon", label="t=%sGyr" %t)
    axes[i].set_xlabel('x [kpc]')
    axes[i].legend()
fig.tight_layout()
plt.show()

# Do it for both galaxies
# Run leapfrog of barnes hut
pos_t, vel_t = leapfrog(ab_pos, ab_vel, masses, softening=softening, N=N, dt=dt)

# Plot evolution of merger
fig,axes = plt.subplots(ncols = 6, nrows=2, figsize=(30, 10))
axes[0][0].quiver(pos_t[0].T[0][:npoints], pos_t[0].T[1][:npoints],
                  vel_t[0].T[0][:npoints], vel_t[0].T[1][:npoints],
                  color="maroon", label="t=0Gyr")
axes[1][0].scatter(pos_t[0].T[0][:npoints], pos_t[0].T[1][:npoints], s=8,
               color="maroon", label="t=0Gyr")
axes[0][0].quiver(pos_t[0].T[0][npoints:], pos_t[0].T[1][npoints:],
                  vel_t[0].T[0][npoints:], vel_t[0].T[1][npoints:],
                  color="green", label="t=0Gyr")
axes[1][0].scatter(pos_t[0].T[0][npoints:], pos_t[0].T[1][npoints:], s=8,
               color="green", label="t=0Gyr")
axes[0][0].set_ylabel('y [kpc]')
axes[1][0].set_ylabel('y [kpc]')
axes[1][0].set_xlabel('x [kpc]')
t_range = np.arange(N/5, N+1, N/5)
for i, t in zip(range(1, 6), t_range):
    t = int(t) - 1
    axes[0][i].quiver(pos_t[t].T[0][:npoints], pos_t[t].T[1][:npoints],
              vel_t[t].T[0][:npoints], vel_t[t].T[1][:npoints],
                   color="maroon", label="t=%sGyr" %t)
    axes[1][i].scatter(pos_t[t].T[0][:npoints], pos_t[t].T[1][:npoints], s=8,
                   color="maroon", label="t=%sGyr" %t)
    axes[0][i].quiver(pos_t[t].T[0][npoints:], pos_t[t].T[1][npoints:],
              vel_t[t].T[0][npoints:], vel_t[t].T[1][npoints:],
                   color="green", label="t=%sGyr" %t)
    axes[1][i].scatter(pos_t[t].T[0][npoints:], pos_t[t].T[1][npoints:], s=8,
                   color="green", label="t=%sGyr" %t)
    axes[1][i].set_xlabel('x [kpc]')
    axes[0][i].legend()
    axes[1][i].legend()
fig.tight_layout()
plt.show()