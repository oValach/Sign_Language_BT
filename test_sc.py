import numpy as np
from scipy import signal, interpolate, spatial
import matplotlib.pyplot as plt
import matplotlib as mpl

y = [1,2,3,4,5,6,7,8,9,10]
z = [1,2,3,4,5,6,7,8,9,10]

a = [1,2,3,4,5,6,7,8,9,10]
b = [2,3,4,5,6,7,8,9,10,22]
c = [20,21,22,23,24,25,26,27,28,40]

plt.style.use('seaborn-notebook')
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(a,y,z, marker='*', color = 'k', linewidth=0, markersize=8)
ax.plot(b,y,z, marker='*', color = 'r', linewidth=0, markersize=8)
ax.plot(c,y,z, marker='*', color = 'g', linewidth=0, markersize=8)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('z')
ax.xaxis.set_tick_params(labelsize=7)
ax.yaxis.set_tick_params(labelsize=7)
ax.zaxis.set_tick_params(labelsize=7)
ax.legend(['A','B', 'C'],loc='best')
plt.show()


correlation = np.corrcoef(a,c)

distances = np.zeros(shape=[len(a)])
for i in range(len(a)):
    distances[i] += spatial.distance.euclidean(a[i],c[i])

print(np.mean(distances))

print(correlation)