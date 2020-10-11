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
npoints=1000
N = 5
dt = 40
# SITUATION 1 Ma:Mb=1:1
M_A = 1e6*u.Msun
M_B = 1e6*u.Msun
# Get initial positions and velocities of the two systems
xyz_A, v_xyz_A = initial_plummer_positions(npoints, M=M_A, seed=798632, radius=r_A, a=a_A, x_pos=-1.2*r_A.value, x_vel=-20)
xyz_B, v_xyz_B = initial_plummer_positions(npoints, M=M_B, seed=987344, radius=r_B, a=a_B, x_pos=1.2*r_B.value, x_vel=20)
# Plot initial conditions
plot_init(xyz_A, xyz_B)
# Run leapfrog of barnes hut
print("Time ellapsed = ", N*dt, "Gyr")
pos_t, vel_t = leapfrog(xyz_A, xyz_B, v_xyz_A, v_xyz_B, M_A=M_A, M_B=M_B, npoints=npoints, N=N, dt=dt)
# Plot spatial evolution of merger
plot_evol(pos_t, npoints, N, dt)

# SITUATION 2 Ma:Mb=1:8
M_A = 1e6*u.Msun
M_B = 8e6*u.Msun
# Get initial positions and velocities of the two systems
xyz_A, v_xyz_A = initial_plummer_positions(npoints, M=M_A, seed=798632, radius=r_A, a=a_A, x_pos=-1.2*r_A.value, x_vel=-20)
xyz_B, v_xyz_B = initial_plummer_positions(npoints, M=M_B, seed=987344, radius=r_B, a=a_B, x_pos=1.2*r_B.value, x_vel=20)
# Plot initial conditions
plot_init(xyz_A, xyz_B)
# Run leapfrog of barnes hut
print("Time ellapsed = ", N*dt, "Gyr")
pos_t, vel_t = leapfrog(xyz_A, xyz_B, v_xyz_A, v_xyz_B, M_A=M_A, M_B=M_B, npoints=npoints, N=N, dt=dt)
# Plot spatial evolution of merger
plot_evol(pos_t, npoints, N, dt)