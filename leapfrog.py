# Primary imports (don't skip)
import numpy as np
from barnes_hut import GravAccel
from initial_conditions import initial_setup
def leapfrog(M_A, M_B, npoints, N=5, dt=2):
    xyz_A, v_xyz_A, xyz_B, v_xyz_B = initial_setup(M_A=M_A, M_B=M_B, npoints=npoints)
    ab_pos = np.concatenate((xyz_A.T, xyz_B.T))
    ab_vel = np.concatenate((v_xyz_A.T, v_xyz_B.T))
    ab_masses = np.repeat((M_A/npoints), len(ab_pos))
    nt = int((N-1)/dt)
    pos_t = np.array([[np.zeros(3) for i in range(len(ab_pos))] for k in range(nt)])
    pos_t[0] = ab_pos
    vel_t = np.array([[np.zeros(3) for i in range(len(ab_vel))] for k in range(nt)])
    vel_t[0] = ab_vel
    mass_t = np.array([np.zeros(len(ab_masses)) for i in range(nt)])
    mass_t[0] = ab_masses
    for t in range(nt-1):# don't need to updte after last point
        accel = GravAccel(pos_t[t], mass_t[t])
    #     kick step: v(i + 1/2) = v(i - 1/2) + a(i) * dt
        vel_t[t+1] = vel_t[t] + accel*dt
    #     drift step: x(i+1) = x(i) + v(i + 1/2) dt
        pos_t[t+1] = pos_t[t] + vel_t[t+1]*dt
    return pos_t, vel_t