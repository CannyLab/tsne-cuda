import itertools
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math

particles = []
n_particles = None
dims = None

min_particle = -1*float("inf")
max_particle =float("inf")



with open('../build/dump_ys.txt') as f:
    line = f.readline().split()
    n_particles = int(line[0])
    dims = float(line[1])
    line = f.readline()
    while line:
        particles.append([float(pos) for pos in line.split()])
        line = f.readline()

particles = np.array(particles)
particles = particles.reshape((-1, n_particles, 2))
n_timesteps = particles.shape[0]

min_particle = np.amin(particles)
max_particle = np.amax(particles)

def animate(i):
    i = i % n_timesteps
    mat.set_data(particles[i,:,0], particles[i,:,1])
    return mat,

fig, ax = plt.subplots()
mat, = ax.plot(particles[0,:,0], particles[0,:,1], 'o', ms=0.6)

ax = plt.axis([min_particle, max_particle, min_particle, max_particle])
ani = animation.FuncAnimation(fig, animate, interval=50)
ani.save('animation.gif', writer='imagemagick', fps=50)
plt.show()