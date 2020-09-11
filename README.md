# Simulating colliding galaxies
A galaxy is a large, self-contained mass of stars. A common shape of a galaxy is a bright center with
spiral arms radiating outward. The sun belongs to the galaxy called the Milky Way. The universe
contains billions of galaxies . With so many galaxies floating around it is clear that sometimes two
galaxies collide, often resulting in two deformed galaxies. Prevalent shapes are a disk, bar, spiral or
ring, for an overview see the Hubble telescope images: http://hubblesite.org/gallery/
album/galaxy.

Project: simulate the collision of two disk-shaped galaxies in 3D. Experiment with various initial
conditions, resulting in realistically shaped galaxies. Galaxies typically comprise 1011 stars. It is
not feasible to do a simulation with that number of stars. Determine experimentally what is feasible.
An important simplification/approximation is the use of the Barnes-Hut method. The essence of this
method is that stars are grouped when they are close together. In this way the complexity of the
algorithm is reduced from N 2 to N.log(N).

Literature:
1. J. Barnes and P. Hut (December 1986). ”A hierarchical O(N log N) force-calculation algorithm”. Nature 324 (4): 446449.
Modelling and Simulation Project: topic to be decided
