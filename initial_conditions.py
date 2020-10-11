# Primary imports (don't skip)
import numpy as np
import random
import astropy.units as u
import astropy.constants as c
import gala
import astropy.units as u
import astropy.constants as c
import gala.dynamics as gd
import gala.potential as gp
from gala.units import galactic
import gala.integrate as gi
from scipy.spatial.transform import Rotation as R
# Initial condition functions
def initial_plummer_positions(npoints, M, seed, radius=15*u.kpc, a=8*u.kpc, x_pos=0,
                              y_pos=0, z_pos=0, x_vel=0, y_vel=0, z_vel=0, G=c.G/c.G.value):
    """
    Input:
    npoints = The number of particles
    M = The mass of the system
    seed = Seed used for random number generation
    radius = The radius of the system
    a = The scale length of the system
    x_pos = The x position of the system
    y_pos = The y position of the system
    z_pos = The z position of the system
    x_vel = The x velocity of the system
    y_vel = The y velocity of the system
    z_vel = The z velocity of the system
    G = The gravitational constant
    
    Finds the initial positions and velocities of a system
    described by a Miyamoto-Nagai profile following Aareseth,
    Henon, Wielen (1974), using the rejection method
    to find velocities within the escpae velocity of the
    system.
    
    Output:
    The arrays of the positions and velocities.
    """
    xyz, v_xyz, rs, vr = [], [], [], []
    np.random.seed(seed)
    # For each particle
    for point in range(npoints):
        # Generate random number and equate it to the enclosed mass to find
        # the radius for the Kuzmin potential
        while True:
            x1 = np.random.uniform(0, M.value)
            r = ((x1/M.value)**(1/3)*a.value)*(1-(x1/M.value)**(2/3))**(-0.5)
            if r < radius.value:
                break
        # Calculate x, y and z and move them onto system
        x3 = np.random.uniform(0, 1)
        z = 0 # setting z=0 for disk
        x = r*np.cos(2*np.pi*x3)
        y = r*np.sin(2*np.pi*x3)
        xyz.append([x+x_pos, y+y_pos, z+z_pos])
        r_shift = np.linalg.norm([x_pos, y_pos, z_pos])
        rs.append(r+r_shift)
        # Escape velocity v=sqrt(-2Phi)
        Ve = np.sqrt(2*G.value*M.value)*(r**2+a.value**2)**(-1/4)
        # Find the x, y and z components of the velocity
        # from the radial velocity
        vz = 0 # Set vz = 0 for the disk
        vx = -Ve*y/r
        vy = Ve*x/r
        # Shift velocity components to the system
        v_xyz.append([vx+x_vel, vy+y_vel, vz+z_vel])
        v_shift = np.linalg.norm([x_vel, y_vel, z_vel])
        vr.append(Ve+v_shift)
    # Correct shape of arrays
    return np.array(xyz).T, np.array(v_xyz).T, rs, vr