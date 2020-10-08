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
def initial_plummer_positions(npoints, M, seed, b=1*u.kpc, radius=15*u.kpc, vgal=600*u.km/u.s, theta=0, phi=0):
    G=c.G
    xyz, v_xyz = [], []
    np.random.seed(seed)
    for point in range(npoints):
        # Generate random x and y in disk
        x = np.random.uniform(-radius.value, radius.value)
        y = np.random.uniform(-radius.value, radius.value)
        z = 0
        r = np.sqrt(x**2+y**2+z**2)
        while r > radius.value:
            r = np.sqrt(x**2+y**2+z**2)
            x = np.random.uniform(-radius.value, radius.value)
            y = np.random.uniform(-radius.value, radius.value)
        # rotate
        xr = x*np.cos(phi)+y*np.sin(phi)*np.cos(theta)+z*np.sin(phi)*np.sin(theta)
        yr = -x*np.sin(phi)+y*np.cos(phi)*np.cos(theta)+z*np.cos(phi)*np.sin(theta)
        zr = -y*np.sin(theta) + z*np.sin(theta)
        # Find the escape velocity and x, y and z using generated
        xyz.append([xr, yr, zr])
        # Use rejection method to generate radial velocity
        # Find the x, y and z velocities from the radial velocity
        Mr = ((u.kpc*r)**3*M*((u.kpc*r)**2+b**2)**(-3/2)).to(u.Msun)
        Ve = (np.sqrt(G*Mr/(r*u.kpc))).to(u.km/u.s)
        vz = 0*u.km/u.s
        vx = -Ve*yr/r
        vy = Ve*xr/r
        v_xyz.append([vx.value, vy.value, vz.value])
    return np.array(xyz).T, np.array(v_xyz).T