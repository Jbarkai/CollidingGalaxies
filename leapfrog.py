# Primary imports (don't skip)
# from pykdgrav import Accel, Potential, BruteForcePotential, BruteForceAccel
from barnes_hut import Accel
import astropy.units as u
import astropy.constants as c

def leapfrog(xyz, v_xyz, masses, softening, N=5, dt=2*u.Gyr, G=c.G):
    """
    Input:
        xyz = The initial positions of the system
        v_xyz = The initial velocities of the system
        M = The mass of the system
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
    # Fill the arrays with initial positions and velocities
    pos_t = [xyz]
    vel_t = [v_xyz]
    dt_in_s = dt.to(u.s).value
    # For each timestep
    for t in range(N-1):# don't need to updte after last point
        # Calculate acceleration with barnes hut method
        accel = Accel(pos_t[t], masses, softening=softening, G=G)
        # kick step: v(i + 1/2) = v(i - 1/2) + a(i) * dt
        vel_t.append(vel_t[t] + accel*dt_in_s)
        # drift step: x(i+1) = x(i) + v(i + 1/2) dt
        pos_t.append(pos_t[t] + vel_t[t+1]*dt_in_s)
    return pos_t, vel_t