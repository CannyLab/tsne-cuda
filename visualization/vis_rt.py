import itertools
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import zmq
import math

print('Waiting to connect...')
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5556")
print('Connected...')

print('Plotting...')
fig, ax = plt.subplots()
axes = plt.gca()
data = np.fromstring(socket.recv(), dtype='float32', count=60000*2).reshape(2,60000)
mat, = ax.plot(data[0], data[1], 'o', ms=0.6)
socket.send(b"hi")
print('Plotted initial.')


def animate(i):
    data = np.fromstring(socket.recv(), dtype='float32', count=60000*2).reshape(2,60000)
    mat.set_data(data[0], data[1])
    socket.send(b"hi")
    axes.set_xlim([np.amin(data[0]),np.amax(data[0])])
    axes.set_ylim([np.amin(data[1]),np.amax(data[1])])
    return mat,

ani = animation.FuncAnimation(fig, animate, interval=1)

print('Anim...')
plt.show()
    

