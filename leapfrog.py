# Primary imports (don't skip)
from pykdgrav import Accel, Potential, BruteForcePotential, BruteForceAccel

def leapfrog(xyz, v_xyz, masses, npoints, N=5, dt=2, G=c.G):
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
    # Setup empty arrays for the length of time
    pos_t = np.array([[np.zeros(3) for i in range(len(xyz))] for k in range(N+1)])
    vel_t = np.array([[np.zeros(3) for i in range(len(v_xyz))] for k in range(N+1)])
    # Fill the arrays with initial positions and velocities
    pos_t[0] = xyz
    vel_t[0] = v_xyz
    # For each timestep
    for t in range(N-1):# don't need to updte after last point
        # Calculate acceleration with barnes hut method
        accel = Accel(pos_t[t], masses, G=G)
        # kick step: v(i + 1/2) = v(i - 1/2) + a(i) * dt
        vel_t[t+1] = vel_t[t] + accel*dt
        # drift step: x(i+1) = x(i) + v(i + 1/2) dt
        pos_t[t+1] = pos_t[t] + vel_t[t+1]*dt
    return pos_t, vel_t