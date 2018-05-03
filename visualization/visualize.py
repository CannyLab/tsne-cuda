import sys
import itertools
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math

MAX_TIMESTEPS = 1000
DIMS = 2

particles = None
n_particles = None

min_particle = -1*float("inf")
max_particle =float("inf")
n_timesteps = 0

filename = '../build/dump_ys.txt'
if len(sys.argv) > 1:
	filename = sys.argv[1]

with open(filename) as f:
    lines = f.readlines()
    n_particles = int(lines[0].split()[0])
    particles = np.array([[float(el) for el in line.split()] for line in lines[1:]])
    n_timesteps = particles.shape[0] // n_particles
    particles = particles[:n_particles * n_timesteps]
    particles = particles.reshape((n_timesteps, n_particles, 2))

# particles = np.array(particles)
# particles = particles.reshape((-1, n_particles, 2))
# n_timesteps = particles.shape[0]

min_particle = np.amin(particles)
max_particle = np.amax(particles)

def animate(i):
    i = i % n_timesteps
    mat.set_data(particles[i,:,0], particles[i,:,1])
    return mat,

fig, ax = plt.subplots()
mat, = ax.plot(particles[0,:,0], particles[0,:,1], 'o', ms=0.6)

ax = plt.axis([min_particle, max_particle, min_particle, max_particle])
ani = animation.FuncAnimation(fig, animate, interval=1)
# ani.save('animation.gif', writer='imagemagick', fps=50)
plt.show()
