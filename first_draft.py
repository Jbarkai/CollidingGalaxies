# Primary imports (don't skip)
import scipy
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.ticker import NullFormatter
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

M_A=1e6*u.Msun
M_B=6e6*u.Msun
npoints=1000

xyz_A, v_xyz_A, xyz_B, v_xyz_B = initial_setup(M_A=M_A, M_B=M_B, npoints=npoints)
plt.style.use('dark_background')
fig,axes = plt.subplots(ncols = 3, figsize=(15,5))
axes[0].scatter(xyz_A[0], xyz_A[1], s=20, marker="*", c="skyblue")
axes[0].scatter(xyz_B[0], xyz_B[1], s=20, marker="*", c="white")
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[1].scatter(xyz_A[0], xyz_A[2], s=20, marker="*", c="skyblue")
axes[1].scatter(xyz_B[0], xyz_B[2], s=20, marker="*", c="white")
# axes[1].set_xlabel('x')
axes[1].set_ylabel('z')
axes[2].scatter(xyz_A[1], xyz_A[2], s=20, marker="*", c="skyblue")
axes[2].scatter(xyz_B[1], xyz_B[2], s=20, marker="*", c="white")
axes[2].set_xlabel('y')
axes[2].set_ylabel('z')
for i in [0, 1, 2]:
    axes[i].set_ylim((-50, 50))
    axes[i].set_xlim((-20, 20))
    axes[i].xaxis.set_major_formatter(NullFormatter())
    axes[i].yaxis.set_major_formatter(NullFormatter())
fig.tight_layout()

class OctNode:
    """Stores the data for an octree node, and spawns its children if possible"""
    def __init__(self, center, size, masses, points, ids, leaves=[]):
        self.center = center                    # center of the node's box
        self.size = size                        # maximum side length of the box
        self.children = []                      # start out assuming that the node has no children
 
        Npoints = len(points)
 
        if Npoints == 1:
            # if we're down to one point, we need to store stuff in the node
            leaves.append(self)
            self.COM = points[0]
            self.mass = masses[0]
            self.id = ids[0]
            self.g = np.zeros(3)        # at each point, we will want the gravitational field
        else:
            self.GenerateChildren(points, masses, ids, leaves)     # if we have at least 2 points in the node,
                                                             # spawn its children
 
            # now we can sum the total mass and center of mass hierarchically, visiting each point once!
            com_total = np.zeros(3) # running total for mass moments to get COM
            m_total = 0.            # running total for masses
            for c in self.children:
                m, com = c.mass, c.COM
                m_total += m
                com_total += com * m   # add the moments of each child
            self.mass = m_total
            self.COM = com_total / self.mass  
 
    def GenerateChildren(self, points, masses, ids, leaves):
        """Generates the node's children"""
        octant_index = (points > self.center)  #does all comparisons needed to determine points' octants
        for i in range(2): #looping over the 8 octants
            for j in range(2):
                for k in range(2):
                    in_octant = np.all(octant_index == np.bool_([i,j,k]), axis=1)
                    if not np.any(in_octant): continue           # if no particles, don't make a node
                    dx = 0.5*self.size*(np.array([i,j,k])-0.5)   # offset between parent and child box centers
                    self.children.append(OctNode(self.center+dx,
                                                 self.size/2,
                                                 masses[in_octant],
                                                 points[in_octant],
                                                 ids[in_octant],
                                                 leaves))
def TreeWalk(node, node0, thetamax=0.7, G=1.0):
    """
    Adds the contribution to the field at node0's point due to particles in node.
    Calling this with the topnode as node will walk the tree to calculate the total field at node0.
    """
    dx = node.COM - node0.COM    # vector between nodes' centres of mass
    r = np.sqrt(np.sum(dx**2))   # distance between them
    if r>0:
        # if the node only has one particle or theta is small enough,
        #  add the field contribution to value stored in node.g
        if (len(node.children)==0) or (node.size/r < thetamax):
            node0.g += G * node.mass * dx/r**3
        else:
            # otherwise split up the node and repeat
            for c in node.children: TreeWalk(c, node0, thetamax, G)
def GravAccel(points, masses, thetamax=0.7, G=1.):
    center = (np.max(points,axis=0)+np.min(points,axis=0))/2       #center of bounding box
    topsize = np.max(np.max(points,axis=0)-np.min(points,axis=0))  #size of bounding box
    leaves = []  # want to keep track of leaf nodes
    topnode = OctNode(center, topsize, masses, points, np.arange(len(masses)), leaves) #build the tree
 
    accel = np.empty_like(points)
    for i,leaf in enumerate(leaves):
        TreeWalk(topnode, leaf, thetamax, G)  # do field summation
        accel[leaf.id] = leaf.g  # get the stored acceleration
 
    return accel

ab_pos = np.concatenate((xyz_A.T.value, xyz_B.T.value))
ab_vel = np.concatenate((v_xyz_A.T.value, v_xyz_B.T.value))
ab_masses = np.repeat((M_A/npoints).value, len(ab_pos))

# p1, v1 = GravAccel(ab_pos, ab_masses, ab_vel)
N = 5
dt = 2
pos_t = np.array([[np.zeros(3) for i in range(len(ab_pos))] for k in range(N-1)])
pos_t[0] = ab_pos
vel_t = np.array([[np.zeros(3) for i in range(len(ab_vel))] for k in range(N-1)])
vel_t[0] = ab_vel
mass_t = np.array([np.zeros(len(ab_masses)) for i in range(N)])
mass_t[0] = ab_masses
for t in range(0,N-1):
    if t < N-2: # don't need to updte after last point
        points = pos_t[t]
        masses = mass_t[t]
        accel = GravAccel(points, masses)
    #     kick step: v(i + 1/2) = v(i - 1/2) + a(i) * dt
        v_half = vel_t[t] - 0.5*accel*dt
        vel_t[t+1] = v_half + accel*dt
    #     drift step: x(i+1) = x(i) + v(i + 1/2) dt
        pos_t[t+1] = pos_t[t] + vel_t[t+1]*dt
        
plt.style.use('dark_background')
fig,axes = plt.subplots(ncols = N-1, figsize=(13,4))
axes[0].scatter(pos_t[0].T[0], pos_t[0].T[1], s=20, marker="*", c="white")
axes[0].set_ylabel('y')
axes[0].xaxis.set_major_formatter(NullFormatter())
axes[0].yaxis.set_major_formatter(NullFormatter())
for t in range(1, N-1):
    axes[t].scatter(pos_t[t].T[0], pos_t[t].T[1], s=20, marker="*", c="white")
    axes[t].set_ylim((-8e6, 8e6))
    axes[t].set_xlim((-4e6, 4e6))
    axes[t].xaxis.set_major_formatter(NullFormatter())
    axes[t].yaxis.set_major_formatter(NullFormatter())
fig.tight_layout()