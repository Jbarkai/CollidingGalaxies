import numpy as np
import astropy.units as u
import astropy.constants as c

def TreeWalk(branch, leaf, theta=0.5, G=c.G.value):
    """
    Input:
        branch = Top branch of the tree to walk through
        leaf = leaf of the branch
        theta = theta criterion
        G = Gravitational constant

    Walks through the tree recursively to caluclate the gravitational
    acceleration at each level
    """        
    s = branch.size # Domain length
    d = branch.COM - leaf.COM # Distance from body to domains COM
    r = np.sqrt(d[0]**2+d[1]**2+d[2]**2) # magnitude of distance
    if r < branch.softening: # add softening
        r = np.sqrt(d[0]**2+d[1]**2+d[2]**2+branch.softening**2) # magnitude of distance
    # Decide if domain is big enough:
    # Theta criterian or only one particle left in domain (no lower branches)
    if (s/r < theta) or (len(branch.subtrees)==0):
        leaf.g += G*branch.mass*d/r**3
    else: # else reject the domain and divide it even smaller
        for subtree in branch.subtrees: TreeWalk(subtree, leaf, theta, G)

class MakeTree:
    def __init__(self, softening, center, length, mass, pos, ids, leaves=[]):
        """
        Input:
            center = The center of the domain
            length = The length of the domain side
            mass = The mass of the particles
            pos = The positions of the particles
            ids = The ids of the nodes
            leaves = The leaves of the tree

        Creates a tree with subtrees below each level if more than one particle
            
        """
        m_tot = 0.
        COM_tot = np.zeros(3)
        self.center = center # center of the domain
        self.size = length # maximum side length of the domain
        self.subtrees = [] # start out assuming this is the only level
        self.id = ids[0]
        self.softening = softening
        # Check if there is only one particle in the domain
        if len(pos) == 1:
            leaves.append(self) # Itself is the last level
            self.COM = pos[0] # The centre of mass is itself
            self.mass = mass[0] # Its own mass is the mass in the domain
            self.g = np.zeros(3) # No gravitational acceleration is nothing since it is alone
        # Else create lower levels of the tree
        else:
            octant_index = (pos > self.center)  #does all comparisons needed to determine points' octants
            # looping through all 8 octants (domains)
            for lev1 in range(2):
                for lev2 in range(2):
                    for lev3 in range(2):
                        # Get the indexes of particles in domain
                        num_particles = np.all(octant_index == np.bool_([lev1,lev2,lev3]), axis=1)
                        # If no particles in domain don't create tree node
                        if not np.any(num_particles):
                            continue
                        m = mass[num_particles]
                        dx = 0.5*self.size*(np.array([lev1,lev2,lev3])-0.5)   # offset between parent and child box centers
                        self.subtrees.append(MakeTree(self.softening,
                                                     self.center+dx,
                                                     self.size/2, # Divide domain in half each time
                                                     m,
                                                     pos[num_particles],
                                                     ids[num_particles],
                                                     leaves))
            # Hierarchically sum total mass and COM
            for subs in self.subtrees:
                m_tot += subs.mass
                COM_tot += subs.COM*subs.mass   # add the moments of each child
            self.mass = m_tot
            self.COM = COM_tot/self.mass

def Accel(pos, mass, softening, theta=0.5, G=c.G):
    """
    Input:
        pos = The initial positions of the particles
        mass = The masses of the particles
        theta = The theta for criterion
        G = The gravitational constant
        
    Builds an oct tree and loops through the leaves to calculate
    the acceleration at leaf

    Output:
        The array of the accelerations in kpc/s^2
    """
    # Spatial extent of first domain that all particles are in
    center = (np.max(pos,axis=0)+np.min(pos,axis=0))/2
    tot_size = np.max(np.max(pos,axis=0)-np.min(pos,axis=0))
    leaves = []  # want to keep track of leaf nodes
    ids = np.arange(len(mass))
    # Build the tree
    first_branch = MakeTree(softening, center, tot_size, mass, pos, ids, leaves)
    accel = np.empty_like(pos)
    units = ((G/G.value)*u.Msun/u.kpc**2).to(u.kpc/u.s**2)
    for i,leaf in enumerate(leaves):
        TreeWalk(first_branch, leaf, theta, G.value)  # do field summation
        accel[leaf.id] = leaf.g*units.value  # get the stored acceleration
    return accel