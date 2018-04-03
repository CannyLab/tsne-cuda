import pickle as pkl

import numpy as np
import matplotlib.pyplot as plt

data = None
with open('results/benchmark_results.pkl', 'rb') as f:
    data = pkl.load(f)

data = {name : np.array(data[name]) for name in data}

sizes = np.array([500, 1000, 2000, 4000, 8000, 16000, 32000])

relative_scaling = sizes * np.log(sizes)
relative_scaling /= relative_scaling[0]

fig1, ax = plt.subplots()
plt.plot(sizes, data['SKLEARN'])
plt.plot(sizes, data['BHTSNE'])
plt.plot(sizes, data['MULTICORE-1'])
plt.plot(sizes, data['BHTSNE'][0] * relative_scaling, '--r')
plt.xlabel('Problem Size (thousands)')
plt.ylabel('Time (s)')
ax.set_xticks(sizes)
ax.set_xticklabels([0.5, 1, 2, 4, 8, 16, 32])
plt.legend(['SKLEARN', 'BHTSNE', 'MULTICORE-1', 'Theoretical NlogN'])

fig2, ax = plt.subplots()
plt.plot(sizes, data['MULTICORE-1'])
plt.plot(sizes, data['MULTICORE-2'])
plt.plot(sizes, data['MULTICORE-3'])
plt.plot(sizes, data['MULTICORE-4'])
plt.xlabel('Problem Size (thousands)')
plt.ylabel('Time (s)')
ax.set_xticks(sizes)
ax.set_xticklabels([0.5, 1, 2, 4, 8, 16, 32])
plt.legend(['MULTICORE-1', 'MULTICORE-2', 'MULTICORE-3', 'MULTICORE-4'])

fig3, ax = plt.subplots()
plt.plot(sizes, data['MULTICORE-1'] / data['MULTICORE-2'])
plt.plot(sizes, data['MULTICORE-1'] / data['MULTICORE-3'])
plt.plot(sizes, data['MULTICORE-1'] / data['MULTICORE-4'])
plt.xlabel('Problem Size (thousands)')
plt.ylabel('Speedup')
ax.set_xticks(sizes)
ax.set_xticklabels([0.5, 1, 2, 4, 8, 16, 32])
plt.legend(['MULTICORE-2', 'MULTICORE-3', 'MULTICORE-4'])

fig1.savefig('results/single-threaded.png')
fig2.savefig('results/multi-threaded.png')
fig3.savefig('results/speedup.png')
plt.show()