# Primary imports (don't skip)
import numpy as np
import random
import astropy.units as u
import astropy.constants as c

# Initial condition functions
def initial_plummer_positions(npoints, M, b, seed):
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
        z = 0
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
    
def initial_plummer_velocities(V, M, b):
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

def initial_setup(M_A, M_B, b_A=1*u.kpc, b_B=1*u.kpc, dt=0.001*u.Gyr, n_steps = 5000, npoints=1000):
    seed_A = 798632
    seed_B = 9397582
    # Get intial positions and velocities of galaxy A
    xyz_A_og, v_xyz_A_og, r_A, vr_A = initial_plummer_positions(npoints, M=M_A, b=b_A, seed=seed_A)
    # Get intial positions and velocities of galaxy B
    xyz_B_og, v_xyz_B_og, r_B, vr_B = initial_plummer_positions(npoints, M=M_B, b=b_B, seed=seed_B)
    # Put them on the MW orbit
    pos_shift = [8.8,0,0.0]
    vel_shift = [150,0,0]
    xyz_A = [np.append(xyz_A_og[i].value, 0) + pos_shift[i] for i in range(3)]*u.kpc
    v_xyz_A = [np.append(v_xyz_A_og[i].value, 0) - vel_shift[i] for i in range(3)]*u.km/u.s
    xyz_B = [np.append(xyz_B_og[i].value, 0) - pos_shift[i] for i in range(3)]*u.kpc
    v_xyz_B = [np.append(v_xyz_B_og[i].value, 0) + vel_shift[i] for i in range(3)]*u.km/u.s
    return xyz_A, v_xyz_A, xyz_B, v_xyz_B