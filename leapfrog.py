# Primary imports (don't skip)
import numpy as np
from pykdgrav import Accel, Potential, BruteForcePotential, BruteForceAccel

def leapfrog(xyz_A, xyz_B, v_xyz_A, v_xyz_B, M_A, M_B, npoints, N=5, dt=2):
    """
    Input:
    xyz_A = The initial positions of system A
    xyz_B = The initial positions of system B
    v_xyz_A = The initial velocities of system A
    v_xyz_B = The initial velocities of system B
    M_A = The mass of system A
    M_B = The mass of system B
    npoints = The number of particles in the system
    N = number of timesteps
    dt = change in step
    
    For each timestep it calculates the gravitational acceleration
    using the barnes hut method and then updates the velocities and
    positions using the Leap Frog method, assuming that the system
    is self-starting (i.e. that v(t=0)=v(t=1/2)).
    Note: N*dt = time ellapsed

    Output:
    The arrays of positions and velocities at each timestep
    """
    # Combine positions, velocities and masses of two systems
    ab_pos = np.concatenate((xyz_A.T, xyz_B.T))[:2*npoints]
    ab_vel = np.concatenate((v_xyz_A.T, v_xyz_B.T))[:2*npoints]
    # Assume the particles are of equal mass
    ab_masses = np.repeat((M_A/npoints), len(ab_pos))
    mass_t = np.array([ab_masses for i in range(N+1)])
    # Setup empty arrays for the length of time
    pos_t = np.array([[np.zeros(3) for i in range(len(ab_pos))] for k in range(N+1)])
    vel_t = np.array([[np.zeros(3) for i in range(len(ab_vel))] for k in range(N+1)])
    # Fill the arrays with initial positions and velocities
    pos_t[0] = ab_pos
    vel_t[0] = ab_vel
    # For each timestep
    for t in range(N-1):# don't need to updte after last point
        # Calculate acceleration with barnes hut method
        accel = Accel(pos_t[t], mass_t[t], G=c.G)
        # kick step: v(i + 1/2) = v(i - 1/2) + a(i) * dt
        vel_t[t+1] = vel_t[t] + accel*dt
        # drift step: x(i+1) = x(i) + v(i + 1/2) dt
        pos_t[t+1] = pos_t[t] + vel_t[t+1]*dt
    return pos_t, vel_t