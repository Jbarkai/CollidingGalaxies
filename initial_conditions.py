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
from scipy.spatial.transform import Rotation as Rotation
# Initial condition functions
def initial_kuzmin_positions(npoints, M, seed, radius=15*u.kpc, a=8*u.kpc, x_pos=0,
                              y_pos=0, z_pos=0, x_vel=0, y_vel=0, z_vel=0, G=c.G):
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
    
    NOTE: positions all in kpc and velocities all in kpc/s
    Finds the initial positions and velocities of a system
    described by a Kuzmin profile following Aareseth,
    Henon, Wielen (1974), using the rejection method
    to find velocities within the escpae velocity of the
    system.
    
    Output:
        The arrays of the positions in kpc and velocities in kpc/s.
    """
    xyz, v_xyz, rs, vr = [], [], [], []
    np.random.seed(seed)
    r_shift = np.linalg.norm([x_pos, y_pos, z_pos])
    # For each particle
    for point in range(npoints):
        # Generate random number and equate it to the enclosed mass to find
        # the radius for the Kuzmin potential
        while True:
            M_enc = np.random.uniform(0, M.value)
            r = ((M_enc/M.value)**(1/3.)*a.value)*(1-(M_enc/M.value)**(2/3.))**(-0.5)
            if r < radius.value:
                break
        # Calculate x, y and z and move them onto system
        theta = np.random.uniform(0, 2*np.pi)
        z = 0 # setting z=0 for disk
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        xyz.append([x+x_pos, y+y_pos, z+z_pos])
        rs.append(r+r_shift)
        # Angular velocity
        V_theta = np.sqrt(G*M_enc*u.Msun/(r*u.kpc)**3).to(1/u.s).value
        # Find the x, y and z components of the velocity
        # Set Radial Velocity to 0
        vz = 0 # Set vz = 0 for the disk
        vx = (-r*V_theta*np.sin(theta))
        vy = (+r*V_theta*np.cos(theta))
        # Shift velocity components to the system
        v_xyz.append([vx+x_vel, vy+y_vel, vz+z_vel])
        Vr = np.linalg.norm([vx, vy, vz])
        v_shift = np.linalg.norm([x_vel, y_vel, z_vel])
        vr.append(Vr+v_shift)
    # Add BH at centre
    # xyz.append([x_pos, y_pos, z_pos])
    # v_xyz.append([x_vel, y_vel, z_vel])
    # rs.append(r_shift)
    # vr.append(np.sqrt(G*(M/2)/(r_shift*u.kpc)).to(u.kpc/u.s).value)
    return np.array(xyz).T, np.array(v_xyz).T, rs, vr