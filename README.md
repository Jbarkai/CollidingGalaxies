# Simulating 2 colliding disk galaxies
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

This code sets up the initial positions and velocities following a Kuzmin potential's mass distribution. For each time step the acceleration is calculated using the barnes-hut method and then the positions and velocities are updated using a leap frog integration.

Literature:
1. J. Barnes and P. Hut (December 1986). ”A hierarchical O(N log N) force-calculation algorithm”. Nature 324 (4): 446449.

## Setup
Package Installation
```
pip install pip
pip install numpy
pip install matplotlib
pip install astropy=3.2.3
pip install astro-gala
pip install scipy==1.2.0
pip install ipyvolume
pip install pytest
pip install random
```

## File Setup
1. `run_merger.py`
The main file in which the simulation is run for three different collision angles.
2. `initial_conditions.py`
This file contains the function which finds the initial positions and velocities of a system described by a Kuzmin profile by first randomly choosing the enclosed mass and using this to find the radial position followed by the cartesian positions and velocities system.
3. `leapfrog.py`
This file integrates the orbit of the nbodies by calculating the gravitational acceleration using the barnes hut method for each timestep and then updating the velocities and positions using the Leap Frog method, assuming that the system is self-starting (i.e. that v(t=0)=v(t=1/2)).
4. `barnes_hut.py`
This file contains the functions needed for the barnes-hut algorithm to calculate the accelerations for each particle.
5. `plotters.py`
This contains all the plotting functions for ease of use.
6. `__init__.py`
This allows for the use of the functions in each of these files.
7. `Videos`
This folder contains the videos for the three simulations. It has two subbfolders, which have the videos in the x-y plane and the x-z plane.
8. `Snapshots`
This folder contains snapshots for each timstep of each simulation, which are used to make the videos. It has two subbfolders, which have the snapshots in the x-y plane and the x-z plane.
9. `Figures`
This folder contains all the figures used in the report, including those displayed in the results.