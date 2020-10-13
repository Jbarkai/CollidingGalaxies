from leapfrog import leapfrog
import scipy
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.ticker import NullFormatter
import astropy.units as u
import astropy.constants as c

# Set initial parameters
r_A = 15*u.kpc
r_B = 15*u.kpc
a_A = 8*u.kpc
a_B = 8*u.kpc
A_vel = 20
B_vel = -20
npoints=1000
N = 50
dt = 0.1
# SITUATION 1 Ma:Mb=1:1
M_A = 1e6*u.Msun
M_B = 1e6*u.Msun
# Get initial positions and velocities of the two systems
xyz_A, v_xyz_A, rs_A, vr_A = initial_plummer_positions(npoints, M=M_A, seed=798632, radius=r_A, a=a_A, x_pos=-1.2*r_A.value, x_vel=A_vel)
xyz_B, v_xyz_B, rs_B, vr_B = initial_plummer_positions(npoints, M=M_B, seed=987344, radius=r_B, a=a_B, x_pos=1.2*r_B.value, x_vel=B_vel)
# Check radial velocity
plt.scatter(rs_A, vr_A, s=8, c='maroon')
plt.xlabel("Radius r [kpc]", fontsize=14)
plt.ylabel(r"Radial Velocity v$_r$ [km/s]", fontsize=14)
plt.show()
# Plot initial positions and velocities
plot_init_vels(xyz_A, xyz_B, v_xyz_A, v_xyz_B)
# Combine positions, velocities and masses of two systems
ab_pos = np.concatenate((xyz_A.T, xyz_B.T))[:2*npoints]
ab_vel = np.concatenate((v_xyz_A.T, v_xyz_B.T))[:2*npoints]
# Assume the particles are of equal mass
#     ab_masses = np.repeat((M_A/npoints), len(ab_pos))
mass_t = np.repeat((M_A/npoints), len(ab_pos))
print("Time ellapsed = ", N*dt, "Gyr")
# Run leapfrog of barnes hut
pos_t, vel_t = leapfrog(ab_pos, ab_vel, mass_t, npoints=npoints, N=N, dt=dt)