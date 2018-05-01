import itertools
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import zmq
import math

print('Waiting to connect...')
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.setsockopt(zmq.SNDTIMEO, 5000)
socket.bind("tcp://*:5556")

num_points = int(socket.recv())
socket.send(b"hi")
print('Connected...')


print('Plotting...')
fig, ax = plt.subplots()
axes = plt.gca()
data = np.fromstring(socket.recv(), dtype='float32', count=num_points*2).reshape(2,num_points)
mat, = ax.plot(data[0], data[1], 'o', ms=0.6)
socket.send(b"hi")
print('Plotted initial.')

def animate(i):
    recv_data = socket.recv()
    if len(recv_data) > 10:
        data = np.fromstring(recv_data, dtype='float32', count=num_points*2).reshape(2,num_points)
        mat.set_data(data[0], data[1])
        axes.set_xlim([np.amin(data[0]),np.amax(data[0])])
        axes.set_ylim([np.amin(data[1]),np.amax(data[1])])
    else:
        num_points = int(recv_data)
    socket.send(b"hi")
    return mat,

ani = animation.FuncAnimation(fig, animate, interval=1)

print('Animating...')
plt.show()






