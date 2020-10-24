from leapfrog import leapfrog
import scipy
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.ticker import NullFormatter
import astropy.units as u
import astropy.constants as c
from plotters import plot_init_vels, plot_evol
from initial_conditions import initial_kuzmin_positions
from scipy.spatial.transform import Rotation as R

# Scenario 1:
## Mass ratio 1:1
## Radii ratio 1:1
## Collision angle 0
# Set initial parameters
r_A = 15*u.kpc
r_B = 15*u.kpc
a_A = 8*u.kpc
a_B = 8*u.kpc
A_vel = 0
B_vel = -0
npoints=1000
tot_npoints = 2*npoints+2
N = 50
dt = 2*u.Gyr
softening = 10*0.98*npoints**(-0.26)
M_A = 1e6*u.Msun
M_B = 1e6*u.Msun
M_BH = 1e6*u.Msun

# Get initial positions and velocities of the two systems
xyz_A, v_xyz_A, rs_A, vr_A = initial_kuzmin_positions(
    npoints, M=M_A, seed=798632, radius=r_A, a=a_A,
    x_pos=-r_A.value/2, z_pos=-0, x_vel=A_vel
)
xyz_B, v_xyz_B, rs_B, vr_B = initial_kuzmin_positions(
    npoints, M=M_B, seed=987344, radius=r_B, a=a_B,
    x_pos=r_B.value/2, x_vel=B_vel
)
# Check radial velocity
plt.scatter(rs_A, (vr_A*u.kpc/u.s).to(u.km/u.s).value, s=8, c='maroon')
plt.xlabel("Radius r [kpc]", fontsize=14)
plt.ylabel(r"Radial Velocity v$_r$ [km/s]", fontsize=14)
plt.show()
# Plot initial conditions
plot_init_vels(xyz_A, xyz_B, v_xyz_A, v_xyz_B, lim=30)
# Assume the particles are of equal mass
masses_A = np.repeat(M_A.value/npoints, npoints)
# Add BH
tot_mass_A = np.append(masses_A, M_BH.value)
tot_xyz_A = np.array(list(xyz_A.T)+ [[-r_A.value/2, 0., 0.]])
tot_v_xyz_A = np.array(list(v_xyz_A.T)+ [[0., 0., 0.]])
# Do it for both galaxies
# Combine positions, velocities and masses of two systems
masses_B = np.repeat(M_B.value/npoints, npoints)
tot_mass_B = np.append(np.repeat(M_B.value/npoints, npoints), M_BH.value)
tot_xyz_B = np.array(list(xyz_B.T)+ [[r_B.value/2, 0., 0.]])
tot_v_xyz_B = np.array(list(v_xyz_B.T)+ [[0., 0., 0.]])
ab_pos = np.concatenate((tot_xyz_A, tot_xyz_B))[:tot_npoints]
ab_vel = np.concatenate((tot_v_xyz_A, tot_v_xyz_B))[:tot_npoints]
masses = np.array(list(tot_mass_A) + list(tot_mass_B))
print("Time ellapsed = ", N*dt)
# Run leapfrog of barnes hut
pos_t, vel_t, accel_t = leapfrog(ab_pos, ab_vel, masses, softening=softening, N=N, dt=dt)
# Plot evolution of merger
plot_evol(pos_t, npoints, N, dt)
#####################################################
# Scenario 2:
## Mass ratio 1:1
## Radii ratio 1:1
## Collision angle pi/4
xyz_A, v_xyz_A, rs_A, vr_A = initial_kuzmin_positions(
    npoints, M=M_A, seed=798632, radius=r_A, a=a_A,
    x_pos=-r_A.value/2, z_pos=-r_A.value/2, x_vel=A_vel
)
rotation = R.from_rotvec(np.array([0, np.pi/4, 0]))
xyz_A = rotation.apply(xyz_A.T).T
v_xyz_A = rotation.apply(v_xyz_A.T).T
# Plot initial conditions
plot_init_vels(xyz_A, xyz_B, v_xyz_A, v_xyz_B, lim=30)
# Assume the particles are of equal mass
masses_A = np.repeat(M_A.value/npoints, npoints)
# Add BH
tot_mass_A = np.append(masses_A, M_BH.value)
tot_xyz_A = np.array(list(xyz_A.T)+ [[-r_A.value/2, -r_A.value/2, 0.]])
tot_v_xyz_A = np.array(list(v_xyz_A.T)+ [[0., 0., 0.]])
# Do it for both galaxies
# Combine positions, velocities and masses of two systems
masses_B = np.repeat(M_B.value/npoints, npoints)
tot_mass_B = np.append(np.repeat(M_B.value/npoints, npoints), M_BH.value)
tot_xyz_B = np.array(list(xyz_B.T)+ [[r_B.value/2, 0., 0.]])
tot_v_xyz_B = np.array(list(v_xyz_B.T)+ [[0., 0., 0.]])
ab_pos = np.concatenate((tot_xyz_A, tot_xyz_B))[:tot_npoints]
ab_vel = np.concatenate((tot_v_xyz_A, tot_v_xyz_B))[:tot_npoints]
masses = np.array(list(tot_mass_A) + list(tot_mass_B))
# Run leapfrog of barnes hut
pos_t, vel_t, accel_t = leapfrog(ab_pos, ab_vel, masses, softening=softening, N=N, dt=dt)
# Plot evolution of merger
plot_evol(pos_t, npoints, N, dt)
#####################################################
# Scenario 3:
## Mass ratio 1:1
## Radii ratio 1:1
## Collision angle pi/2
xyz_B, v_xyz_B, rs_B, vr_B = initial_kuzmin_positions(
    npoints, M=M_B, seed=987344, radius=r_B, a=a_B,
    x_pos=r_B.value/2, z_pos=r_A.value/2, x_vel=B_vel
)
# Rotate them
rotation2 = R.from_rotvec(np.array([0, -np.pi/4, 0]))
xyz_B = rotation2.apply(xyz_B.T).T
v_xyz_B = rotation2.apply(v_xyz_B.T).T
# Plot initial conditions
plot_init_vels(xyz_A, xyz_B, v_xyz_A, v_xyz_B, lim=30)
# Assume the particles are of equal mass
masses_A = np.repeat(M_A.value/npoints, npoints)
# Add BH
tot_mass_A = np.append(masses_A, M_BH.value)
tot_xyz_A = np.array(list(xyz_A.T)+ [[-r_A.value/2, -r_A.value/2, 0.]])
tot_v_xyz_A = np.array(list(v_xyz_A.T)+ [[0., 0., 0.]])
# Do it for both galaxies
# Combine positions, velocities and masses of two systems
masses_B = np.repeat(M_B.value/npoints, npoints)
tot_mass_B = np.append(np.repeat(M_B.value/npoints, npoints), M_BH.value)
tot_xyz_B = np.array(list(xyz_B.T)+ [[r_B.value/2, 0., 0.]])
tot_v_xyz_B = np.array(list(v_xyz_B.T)+ [[0., 0., 0.]])
ab_pos = np.concatenate((tot_xyz_A, tot_xyz_B))[:tot_npoints]
ab_vel = np.concatenate((tot_v_xyz_A, tot_v_xyz_B))[:tot_npoints]
masses = np.array(list(tot_mass_A) + list(tot_mass_B))
# Run leapfrog of barnes hut
pos_t, vel_t, accel_t = leapfrog(ab_pos, ab_vel, masses, softening=softening, N=N, dt=dt)
# Plot evolution of merger
plot_evol(pos_t, npoints, N, dt)
#####################################################
# Scenario 4
## Mass ratio 1:1
## Radii ratio 1:3
## Collision angle pi/2
r_B = 5*u.kpc
xyz_B, v_xyz_B, rs_B, vr_B = initial_kuzmin_positions(
    npoints, M=M_B, seed=987344, radius=r_B, a=a_B,
    x_pos=r_B.value/2, z_pos=r_A.value/2, x_vel=B_vel
)
# Rotate them
xyz_B = rotation2.apply(xyz_B.T).T
v_xyz_B = rotation2.apply(v_xyz_B.T).T
# Plot initial conditions
plot_init_vels(xyz_A, xyz_B, v_xyz_A, v_xyz_B, lim=30)
# Assume the particles are of equal mass
masses_A = np.repeat(M_A.value/npoints, npoints)
# Add BH
tot_mass_A = np.append(masses_A, M_BH.value)
tot_xyz_A = np.array(list(xyz_A.T)+ [[-r_A.value/2, -r_A.value/2, 0.]])
tot_v_xyz_A = np.array(list(v_xyz_A.T)+ [[0., 0., 0.]])
# Do it for both galaxies
# Combine positions, velocities and masses of two systems
masses_B = np.repeat(M_B.value/npoints, npoints)
tot_mass_B = np.append(np.repeat(M_B.value/npoints, npoints), M_BH.value)
tot_xyz_B = np.array(list(xyz_B.T)+ [[r_B.value/2, 0., 0.]])
tot_v_xyz_B = np.array(list(v_xyz_B.T)+ [[0., 0., 0.]])
ab_pos = np.concatenate((tot_xyz_A, tot_xyz_B))[:tot_npoints]
ab_vel = np.concatenate((tot_v_xyz_A, tot_v_xyz_B))[:tot_npoints]
masses = np.array(list(tot_mass_A) + list(tot_mass_B))
# Run leapfrog of barnes hut
pos_t, vel_t, accel_t = leapfrog(ab_pos, ab_vel, masses, softening=softening, N=N, dt=dt)
# Plot evolution of merger
plot_evol(pos_t, npoints, N, dt)
#####################################################
# Scenario 5
## Mass ratio 1:6
## Radii ratio 1:1
## Collision angle pi/2
r_B = 15*u.kpc
M_A = 6e6*u.Msun
# Get initial positions and velocities of the two systems
xyz_A, v_xyz_A, rs_A, vr_A = initial_kuzmin_positions(
    npoints, M=M_A, seed=798632, radius=r_A, a=a_A,
    x_pos=-r_A.value/2, z_pos=-r_A.value/2, x_vel=A_vel
)
xyz_B, v_xyz_B, rs_B, vr_B = initial_kuzmin_positions(
    npoints, M=M_B, seed=987344, radius=r_B, a=a_B,
    x_pos=r_B.value/2, z_pos=r_A.value/2, x_vel=B_vel
)
# Rotate them
rotation1 = R.from_rotvec(np.array([0, np.pi/4, 0]))
xyz_A = rotation1.apply(xyz_A.T).T
v_xyz_A = rotation1.apply(v_xyz_A.T).T
rotation2 = R.from_rotvec(np.array([0, -np.pi/4, 0]))
xyz_B = rotation2.apply(xyz_B.T).T
v_xyz_B = rotation2.apply(v_xyz_B.T).T
# Plot initial conditions
plot_init_vels(xyz_A, xyz_B, v_xyz_A, v_xyz_B, lim=30)
# Assume the particles are of equal mass
masses_A = np.repeat(M_A.value/npoints, npoints)
# Add BH
tot_mass_A = np.append(masses_A, 6*M_BH.value)
tot_xyz_A = np.array(list(xyz_A.T)+ [[-r_A.value/2, -r_A.value/2, 0.]])
tot_v_xyz_A = np.array(list(v_xyz_A.T)+ [[0., 0., 0.]])
# Do it for both galaxies
# Combine positions, velocities and masses of two systems
masses_B = np.repeat(M_B.value/npoints, npoints)
tot_mass_B = np.append(np.repeat(M_B.value/npoints, npoints), M_BH.value)
tot_xyz_B = np.array(list(xyz_B.T)+ [[r_B.value/2, 0., 0.]])
tot_v_xyz_B = np.array(list(v_xyz_B.T)+ [[0., 0., 0.]])
ab_pos = np.concatenate((tot_xyz_A, tot_xyz_B))[:tot_npoints]
ab_vel = np.concatenate((tot_v_xyz_A, tot_v_xyz_B))[:tot_npoints]
masses = np.array(list(tot_mass_A) + list(tot_mass_B))
# Run leapfrog of barnes hut
pos_t, vel_t, accel_t = leapfrog(ab_pos, ab_vel, masses, softening=softening, N=N, dt=dt)
# Plot evolution of merger
plot_evol(pos_t, npoints, N, dt)