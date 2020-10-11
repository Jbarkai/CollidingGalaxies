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
                              y_pos=0, z_pos=0, x_vel=0, y_vel=0, z_vel=0):
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
    
    Finds the initial positions and velocities of a system
    described by a Miyamoto-Nagai profile following Aareseth,
    Henon, Wielen (1974), using the rejection method
    to find velocities within the escpae velocity of the
    system.
    
    Output:
    The arrays of the positions and velocities.
    """
    G=c.G
    xyz, v_xyz = [], []
    np.random.seed(seed)
    xyz, v_xys = [], []
    np.random.seed(seed)
    # Energy of the system
    E = ((3*np.pi/64)*(G*M**2/radius)).to(u.kpc**2*u.Msun/u.s**2)
    # velocity scaler
    v_scale = ((64/3*np.pi)*(E**0.5/M**0.5)).to(u.km/u.s)
    # For each particle
    for point in range(npoints):
        # Generate random number and equate it to the enclosed mass to find
        # the radius for the Kuzmin potential
        while True:
            x1 = np.random.uniform(0, 1)
            r = (x1**(1/3)*a.value)*(1-x1**(2/3))**(-0.5)
            if r < 1:
                break
        # Calculate x, y and z and move them onto system
        x3 = np.random.uniform(0, 1)
        z = 0 # setting z=0 for disk
        x = r*np.cos(2*np.pi*x3)*radius.value
        y = r*np.sin(2*np.pi*x3)*radius.value
        xyz.append([x+x_pos, y+y_pos, z+z_pos])
        # Use rejection method to generate radial velocity
        # that is within the escape velocity
        x5 = np.random.uniform(0, 1)
        x4 = np.random.uniform(0, 1)
        g = x4**2*(1-x4**2)**(7/2)
        while 0.1*x5 > g:
            x4 = np.random.uniform(0, 1)
            g = x4**2*(1-x4**2)**(7/2)
            x5 = np.random.uniform(0, 1)
        # Escape velocity
        Ve = np.sqrt(x1/r)
        # Find the x, y and z velocities from the radial velocity
        V = x4*Ve
        # Find the x, y and z components of the velocity
        # from the radial velocity
        x7 = np.random.uniform(0, 1)
        vz = 0*v_scale.value # Set vz = 0 for the disk
        vx = V*np.cos(2*np.pi*x7)*v_scale.value
        vy = V*np.sin(2*np.pi*x7)*v_scale.value
        # Scale velocity components to the system
        v_xyz.append([vx+x_vel, vy+y_vel, vz+z_vel])
    # Correct shape of arrays
    return np.array(xyz).T, np.array(v_xyz).T