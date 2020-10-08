# Primary imports (don't skip)
import numpy as np
from barnes_hut import GravAccel
from pykdgrav import Accel, Potential, BruteForcePotential, BruteForceAccel

def leapfrog(xyz_A, xyz_B, v_xyz_A, v_xyz_B, M_A, M_B, npoints, N=5, dt=2):
    # N = number of timesteps
    # dt = change in step
    # N*dt = time ellapsed
    ab_pos = np.concatenate((xyz_A.T, xyz_B.T))[:2*npoints]
    ab_vel = np.concatenate((v_xyz_A.T, v_xyz_B.T))[:2*npoints]
    ab_masses = np.repeat((M_A/npoints), len(ab_pos))
    pos_t = np.array([[np.zeros(3) for i in range(len(ab_pos))] for k in range(N+1)])
    pos_t[0] = ab_pos
    vel_t = np.array([[np.zeros(3) for i in range(len(ab_vel))] for k in range(N+1)])
    vel_t[0] = ab_vel
    mass_t = np.array([ab_masses for i in range(N+1)])
    for t in range(N-1):# don't need to updte after last point
        accel = Accel(pos_t[t], mass_t[t], G=c.G)
    #     kick step: v(i + 1/2) = v(i - 1/2) + a(i) * dt
        vel_t[t+1] = vel_t[t] + accel*dt
    #     drift step: x(i+1) = x(i) + v(i + 1/2) dt
        pos_t[t+1] = pos_t[t] + vel_t[t+1]*dt
    return pos_t, vel_t