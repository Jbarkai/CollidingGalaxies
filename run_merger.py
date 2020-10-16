from leapfrog import leapfrog
import scipy
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.ticker import NullFormatter
import astropy.units as u
import astropy.constants as c
from plotters import plot_init, plot_evol, plot_init_vels, plot_evol_vels
from initial_conditions import initial_kuzmin_positions

# Set initial parameters
r_A = 30*u.kpc
r_B = 30*u.kpc
a_A = 8*u.kpc
a_B = 8*u.kpc
A_vel = 0
B_vel = -0
npoints=100
tot_npoints = 2*npoints+2
N = 5
dt = 1*u.Gyr
softening = 0.98*npoints**(-0.26)
# SITUATION 1 Ma:Mb=1:1
M_A = 1e6*u.Msun
M_B = 1e6*u.Msun

M_BH = M_A*1e3/npoints
# Get initial positions and velocities of the two systems
xyz_A, v_xyz_A, rs_A, vr_A = initial_kuzmin_positions(
    npoints, M=M_A, seed=798632, radius=r_A, a=a_A,
    x_pos=0, z_pos=-0, x_vel=A_vel
)
xyz_B, v_xyz_B, rs_B, vr_B = initial_kuzmin_positions(
    npoints, M=M_B, seed=987344, radius=r_B, a=a_B,
    x_pos=r_B.value, x_vel=B_vel
)
# Check radial velocity
plt.scatter(rs_A, (vr_A*u.kpc/u.s).to(u.km/u.s).value, s=8, c='maroon')
plt.xlabel("Radius r [kpc]", fontsize=14)
plt.ylabel(r"Radial Velocity v$_r$ [km/s]", fontsize=14)
plt.show()
# Plot initial conditions
plot_init_vels(xyz_A, xyz_B, v_xyz_A, v_xyz_B, lim=50)

# Assume the particles are of equal mass
masses_A = np.repeat(M_A.value/npoints, npoints)
# Add BH
tot_mass_A = np.append(masses_A, M_BH.value)
tot_xyz_A = np.array(list(xyz_A.T)+ [[0., 0., 0.]])
tot_v_xyz_A = np.array(list(v_xyz_A.T)+ [[0., 0., 0.]])
print("Time ellapsed = ", N*dt)
# Test integration on one galaxy
# Run leapfrog of barnes hut
pos_t_test, vel_t_test, accel_test = leapfrog(tot_xyz_A, tot_v_xyz_A, tot_mass_A, softening=softening, N=N, dt=dt)


# Plot evolution of merger
fig,axes = plt.subplots(ncols = 5, figsize=(20, 4))
axes[0].set_ylabel('y [kpc]')
t_range = np.arange(N/5, N+1, N/5)
for i, t in zip(range(1, 6), t_range):
    t = int(t) - 1
    axes[i-1].quiver(pos_t_test[t].T[0], pos_t_test[t].T[1],
              vel_t_test[t].T[0], vel_t_test[t].T[1],
                   color="maroon", label="t=%sGyr" %t)
    axes[i-1].set_xlabel('x [kpc]')
    axes[i-1].legend()
fig.tight_layout()
plt.show()