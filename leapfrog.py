# Primary imports (don't skip)
from pykdgrav import Accel, Potential, BruteForcePotential, BruteForceAccel

def leapfrog(xyz, v_xyz, M, npoints, N=5, dt=2, G=1):
    """
    Input:
    xyz = The initial positions of the system
    v_xyz = The initial velocities of the system
    M = The mass of the system
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
    # For each timestep
    for t in range(N-1):# don't need to updte after last point
        # Calculate acceleration with barnes hut method
        accel = Accel(pos_t[t], mass_t[t], G=G)
        # kick step: v(i + 1/2) = v(i - 1/2) + a(i) * dt
        vel_t[t+1] = vel_t[t] + accel*dt
        # drift step: x(i+1) = x(i) + v(i + 1/2) dt
        pos_t[t+1] = pos_t[t] + vel_t[t+1]*dt
    return pos_t, vel_t