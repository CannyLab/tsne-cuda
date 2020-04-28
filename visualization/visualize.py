import sys
import itertools
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import tqdm
matplotlib.use('Agg')


MAX_TIMESTEPS = 1000
DIMS = 2

particles = None
n_particles = None

min_particle = -1*float("inf")
max_particle = float("inf")
n_timesteps = 0

filename = '../build/dump_ys.txt'
if len(sys.argv) > 1:
    filename = sys.argv[1]

with open(filename) as f:

    n_particles = int(f.readline().split()[0])

    particles_list = []
    for idx, line in enumerate(tqdm.tqdm(f)):
        if idx % 20 != 0:
            continue
        particles_list.append([float(el) for el in line.split()])

    particles = np.array(particles_list)
    n_timesteps = particles.shape[0] // n_particles
    particles = particles[:n_particles * n_timesteps]
    particles = particles.reshape((n_timesteps, n_particles, 2))
    print(particles.shape)

# particles = np.array(particles)
# particles = particles.reshape((-1, n_particles, 2))
# n_timesteps = particles.shape[0]

min_particle = np.amin(particles)
max_particle = np.amax(particles)

print(min_particle)
print(max_particle)


def animate(i):
    i = i % n_timesteps
    mat.set_data(particles[i, :, 0], particles[i, :, 1])
    return mat,


fig, ax = plt.subplots()
mat, = ax.plot(particles[0, :, 0], particles[0, :, 1], 'o', ms=0.6)

ax = plt.axis([min_particle, max_particle, min_particle, max_particle])
ani = animation.FuncAnimation(fig, animate, interval=1)
ani.save('animation.gif', writer='imagemagick', fps=50)
