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
theta_A = 45
theta_B = 0
phi_A = 30
phi_B = 0
npoints=1000
N = 5
dt = 40
print("Time ellapsed = ", N*dt, "Gyr")
M_A = 1e6*u.Msun
M_B = 1e6*u.Msun
# SITUATION 1 Ma:Mb=1:1
# Get initial positions and velocities
xyz_A, v_xyz_A = initial_plummer_positions(npoints, M=M_A, seed=798632, theta=theta_A, phi=phi_A, radius=r_A)
xyz_B, v_xyz_B = initial_plummer_positions(npoints, M=M_B, seed=294897, theta=theta_B, phi=phi_B, radius=r_B)
# Shift the galaxies away from each other
pos_A = [-1.2*r_A.value, 0, 0]
pos_B = [1.2*r_B.value, 0, 0]
A_xyz = [np.append(xyz_A[i], 0) + pos_A[i] for i in range(3)]*u.kpc
B_xyz = [np.append(xyz_B[i], 0) + pos_B[i] for i in range(3)]*u.kpc
# Plot initial conditions
plot_init(A_xyz, B_xyz)
# Run leapfrog of barnes hut
pos_t, vel_t = leapfrog(A_xyz, B_xyz, v_xyz_A, v_xyz_B, M_A=M_A, M_B=M_B, npoints=npoints, N=N, dt=dt)
# Plot spatial evolution of merger
plot_evol(pos_t, npoints, N, dt)


# SITUATION 2 Ma:Mb=1:8
M_A = 1e6*u.Msun
M_B = 8e6*u.Msun

# Get initial positions and velocities
xyz_A, v_xyz_A = initial_plummer_positions(npoints, M=M_A, seed=798632, theta=theta_A, phi=phi_A, radius=r_A)
xyz_B, v_xyz_B = initial_plummer_positions(npoints, M=M_B, seed=294897, theta=theta_B, phi=phi_B, radius=r_B)
# Shift the galaxies away from each other
pos_A = [-1.2*r_A.value, 0, 0]
pos_B = [1.2*r_B.value, 0, 0]
A_xyz = [np.append(xyz_A[i], 0) + pos_A[i] for i in range(3)]*u.kpc
B_xyz = [np.append(xyz_B[i], 0) + pos_B[i] for i in range(3)]*u.kpc
# Plot initial conditions
plot_init(A_xyz, B_xyz)
# Run leapfrog of barnes hut
pos_t, vel_t = leapfrog(A_xyz, B_xyz, v_xyz_A, v_xyz_B, M_A=M_A, M_B=M_B, npoints=npoints, N=N, dt=dt)
# Plot spatial evolution of merger
plot_evol(pos_t, npoints, N, dt)