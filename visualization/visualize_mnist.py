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

labels = np.fromfile('../train-labels.idx1-ubyte', dtype=np.uint8)
labels = labels[8:].astype(np.int32)

with open('../build/dump_ys.txt') as f:
    line = f.readline().split()
    n_particles = int(line[0])
    dims = float(line[1])
    line = f.readline()
    while line:
        particles.append([float(pos) for pos in line.split()])
        line = f.readline()

particles = np.array(particles)

print(particles.shape)
particles = particles.reshape((-1, n_particles, 2))
print(particles.shape)
n_timesteps = particles.shape[0]

min_particle = np.amin(particles)
max_particle = np.amax(particles)

print(particles[0].shape)
print(particles[0,:,0].shape)

fig, ax = plt.subplots()
mat = ax.scatter(particles[-1,:,0], particles[-1,:,0], s=0.6,c=labels,cmap='tab10')

# def animate(i):
#     i = i % n_timesteps
#     mat.set_offsets(particles[i])
#     return mat,

ax = plt.axis([min_particle, max_particle, min_particle, max_particle])
ani = animation.FuncAnimation(fig, animate, interval=1)
# ani.save('animation.gif', writer='imagemagick', fps=50)
plt.show()
