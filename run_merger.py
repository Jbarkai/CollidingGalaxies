from leapfrog import leapfrog
import scipy
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.ticker import NullFormatter
import astropy.units as u
import astropy.constants as c
import ipyvolume as ipv


M_A=1e6*u.Msun
M_B=6e6*u.Msun
npoints=1000
N = 6
dt = 0.5
pos_t, vel_t = leapfrog(M_A=1e6*u.Msun, M_B=1e6*u.Msun, npoints=1000, N=N, dt=dt)
# plt.style.use('dark_background')
fig,axes = plt.subplots(ncols = N-1, nrows=2, figsize=(13,4))
axes[0][0].scatter(pos_t[0].T[0][:1001], pos_t[0].T[1][:1001], s=20, marker="*", c="white")
axes[0][0].scatter(pos_t[0].T[0][1001:], pos_t[0].T[1][1001:], s=20, marker="*", c="skyblue")
axes[0][0].set_ylabel('y [kpc]')
# axes[0][0].xaxis.set_major_formatter(NullFormatter())
# axes[0][0].yaxis.set_major_formatter(NullFormatter())
axes[1][0].scatter(pos_t[0].T[0][:1001], pos_t[0].T[2][:1001], s=20, marker="*", c="white")
axes[1][0].scatter(pos_t[0].T[0][1001:], pos_t[0].T[2][1001:], s=20, marker="*", c="skyblue")
axes[1][0].set_ylabel('z [kpc]')
axes[1][0].set_xlabel('x [kpc]')
# axes[1][0].xaxis.set_major_formatter(NullFormatter())
# axes[1][0].yaxis.set_major_formatter(NullFormatter())
# axes[0][0].set_ylim((-20, 20))
# axes[0][0].set_xlim((-20, 20))
t_range = np.arange(1, (N-1)/dt, ((N-1)/dt)/5)
for i, t in zip(range(1, 5), t_range):
    t = int(t)
    for k in range(2):
        axes[k][i].scatter(pos_t[t].T[0][1001:], pos_t[t].T[k+1][1001:], s=20, marker="*", c="skyblue")
        axes[k][i].scatter(pos_t[t].T[0][:1001], pos_t[t].T[k+1][:1001], s=20, marker="*", c="white")
        axes[1][i].set_xlabel('x [kpc]')
        axes[k][i].set_ylim((-i*1e4, i*1e4))
        axes[k][i].set_xlim((-i*1e4, i*1e4))
#         axes[k][i].xaxis.set_major_formatter(NullFormatter())
#         axes[k][i].yaxis.set_major_formatter(NullFormatter())
fig.tight_layout(w_pad=-2)

# fig = ipv.figure(width=800, height=500)
# q = ipv.quiver(pos_t.T[0], pos_t.T[1], pos_t.T[2], vel_t.T[0], vel_t.T[1], vel_t.T[2], color="red", size=7)

# ipv.xyzlim(-15,15)
# ipv.xlabel('x [kpc]')
# ipv.ylabel('y [kpc]')
# ipv.zlabel('z [kpc]')

# ipv.show()