# Primary imports (don't skip)
import scipy
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
import ipyvolume as ipv
from matplotlib.ticker import NullFormatter
# Secondary imports (can skip)

# Wider notebook:
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

# Fancy figures
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
%config IPCompleter.use_jedi = False 

try:
    plt.style.use('bmh')
except:
    pass

mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['lines.markersize'] = 10
# Some additional imports
import gala
import astropy.units as u
import astropy.constants as c
import gala.dynamics as gd
import gala.potential as gp
from gala.units import galactic
import gala.integrate as gi
import astropy.coordinates as coord

# Initial condition functions
def initial_plummer_positions(npoints=1000, M=1e6*u.Msun, b=1*u.kpc, seed=798632):
    """
    Input:
    npoints = The number of particles
    M = The mass of the system
    b = The scale length of the system
    seed = Seed used for random number generation
    
    Finds the initial positions and velocities of a system
    described by a plummer profile following Aareseth,
    Henon, Wielen (1974), using the rejection method
    to find velocities within the escpae velocity of the
    system.
    
    Output:
    The arrays of the positions and velocities as well
    as the radii.
    """
    G=c.G
    x_f, y_f, z_f, r_f, vx_f, vy_f, vz_f, vr_f = [], [], [], [], [], [], [], []
    np.random.seed(seed)
    for point in range(npoints):
        # Generate random number and equate it to the mass to find
        # the radius
        x1 = np.random.uniform(0, 1)
        x2 = np.random.uniform(0, 1)
        x3 = np.random.uniform(0, 1)
        r = (x1**(-2/3)-1)**(-0.5)
        # Find the escape velocity and x, y and z using generated
        # radius
        Ve = 2**(0.5)*(1+r**2)**(-1/4)
#         z = (1-2*x2)*r
        z=0
        x = (r**2-z**2)**(0.5)*np.cos(2*np.pi*x3)
        y = (r**2-z**2)**(0.5)*np.sin(2*np.pi*x3)
        x_f.append(x)
        y_f.append(y)
        z_f.append(z)
        r_f.append(r)
        # Use rejection method to generate radial velocity
        # that is within the escape velocity
        x5 = np.random.uniform(0, 1)
        x4 = np.random.uniform(0, 1)
        g = x4**2*(1-x4**2)**(7/2)
        while 0.1*x5 > g:
            x4 = np.random.uniform(0, 1)
            g = x4**2*(1-x4**2)**(7/2)
            x5 = np.random.uniform(0, 1)
        # Find the x, y and z velocities from the radial velocity
        V = x4*Ve
        vx, vy, vz, Vr = initial_plummer_velocities(V, M=M, b=b)
        vx_f.append(vx)
        vy_f.append(vy)
        vz_f.append(vz)
        vr_f.append(Vr)
    xyz = [x_f, y_f, z_f]
    v_xyz = [vx_f, vy_f, vz_f]*u.km/u.s
    # Scale position components to the system
    E = 3*np.pi*G*M**2/(64*b)
    xyz_scale = 3*np.pi*M**2/(64*E)
    xyz = xyz*xyz_scale*G
    r_f = r_f*xyz_scale*G
    return xyz, v_xyz, r_f, vr_f
    
def initial_plummer_velocities(V, M=1e6*u.Msun, b=1*u.kpc):
    """
    Input:
    V = Radial velocity boundary
    M = The mass of the system
    b = The scale length of the system
    
    Finds the initial velocity of a particle in a
    system described by a plummer profile from
    its radial velocity following Aareseth, Henon,
    Wielen (1974).
    
    Output:
    The x, y and z components of the velocity.
    """
    G=c.G
    # Find the x, y and z components of the velocity
    # from the radial velocity
    x6 = np.random.uniform(0, 1)
    x7 = np.random.uniform(0, 1)
    vz = (1-2*x6)*V
    vx = (V**2-vz**2)**(0.5)*np.cos(2*np.pi*x7)
    vy = (V**2-vz**2)**(0.5)*np.sin(2*np.pi*x7)
    # Scale velocity components to the system
    E = 3*np.pi*G*M**2/(64*b)
    scale = 64*E**(0.5)/(3*np.pi*M**(0.5))
    vx = vx*scale.to(u.km/u.s)
    vy = vy*scale.to(u.km/u.s)
    vz = vz*scale.to(u.km/u.s)
    V = V*scale.to(u.km/u.s)
    return vx.value, vy.value, vz.value, V.value

# Set up galactuc units
un = gala.units.galactic
# Choose number of points and size of steps
# for integration as well as integrator
npoints=1000
dt=0.001*u.Gyr
integ = gi.LeapfrogIntegrator
# Get intial positions and velocities of galaxy A
xyz_A, v_xyz_A, r_A, vr_A = initial_plummer_positions(npoints, seed=798632)
# Get intial positions and velocities of galaxy B
xyz_B, v_xyz_B, r_B, vr_B = initial_plummer_positions(npoints, seed=9397582)
# Set up the potentials used
MW_potential = gp.MilkyWayPotential(units=un)
# Set up the Hamiltonians for the various potentials used
MW_Hamiltonian = gp.Hamiltonian(MW_potential)
w0_A = gd.PhaseSpacePosition(pos=xyz_A, vel=v_xyz_A)
w0_B = gd.PhaseSpacePosition(pos=xyz_B, vel=v_xyz_B)
orbit_A = MW_Hamiltonian.integrate_orbit(w0_A, n_steps=5000, dt=dt, Integrator=integ)
orbit_B = MW_Hamiltonian.integrate_orbit(w0_B, n_steps=5000, dt=dt, Integrator=integ)
new_init_A_xyz = [orbit_A.x[-1], orbit_A.y[-1], orbit_A.z[-1]]
new_init_B_xyz = [orbit_B.x[-1], orbit_B.y[-1], orbit_B.z[-1]]
new_init_A_v_xyz = [orbit_A.v_x[-1], orbit_A.v_y[-1], orbit_A.v_z[-1]]
new_init_B_v_xyz = [orbit_B.v_x[-1], orbit_B.v_y[-1], orbit_B.v_z[-1]]

# Positions of galaxy A and B
pos_A = [-80, 0, 0]
vel_A = [30, 30, 0]
pos_B = [30, 0, 0]
vel_B = [-80,80,0]
new_init_A_xyz = [np.append(new_init_A_xyz[i].value, 0) + pos_A[i] for i in range(3)]*u.kpc
new_init_A_xyz = [np.append(new_init_A_xyz[i].value, 0) + vel_A[i] for i in range(3)]*u.km/u.s
new_init_B_xyz = [np.append(new_init_B_xyz[i].value, 0) + pos_B[i] for i in range(3)]*u.kpc
new_init_B_vxyz = [np.append(new_init_B_vxyz[i].value, 0) + vel_B[i] for i in range(3)]*u.km/u.s
plt.style.use('dark_background')
fig,axes = plt.subplots(ncols = 3, figsize=(15,5))
axes[0].scatter(new_init_A_xyz[0], new_init_A_xyz[1], s=20, marker="*", c="skyblue")
axes[0].scatter(new_init_B_xyz[0], new_init_B_xyz[1], s=20, marker="*", c="white")
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[1].scatter(new_init_A_xyz[0], new_init_A_xyz[2], s=20, marker="*", c="skyblue")
axes[1].scatter(new_init_B_xyz[0], new_init_B_xyz[2], s=20, marker="*", c="white")
axes[1].set_xlabel('x')
axes[1].set_ylabel('z')
axes[2].scatter(new_init_A_xyz[1], new_init_A_xyz[2], s=20, marker="*", c="skyblue")
axes[2].scatter(new_init_B_xyz[1], new_init_B_xyz[2], s=20, marker="*", c="white")
axes[2].set_xlabel('y')
axes[2].set_ylabel('z')
for i in [0, 1, 2]:
    axes[i].set_xlim((-100, 100))
    axes[i].xaxis.set_major_formatter(NullFormatter())
    axes[i].yaxis.set_major_formatter(NullFormatter())
fig.tight_layout()