import matplotlib.pyplot as plt
from matplotlib.text import get_rotation
import numpy as np

"""
x1 = [3,2,4,1,1,1]
x2 = [5,4,1,2,2,2]
x3 = [1,5,3,3,4,3]
x4 = [4,3,2,5,3,4]
x5 = [2,1,5,4,5,5]
"""
x1 = [5,1,1,1,1,1]
x2 = [2,2,2,2,2,3]
x3 = [1,3,4,3,5,4]
x4 = [4,5,3,4,3,2]
x5 = [3,4,5,5,4,5]

y = ['1','3','5','10','20','30']

plt.style.use('seaborn')
plt.figure()
plt.plot(y,x1,  color = 'limegreen', linestyle='--', marker='o', markersize=7, linewidth=0.9, markerfacecolor='limegreen',markeredgewidth=1, markeredgecolor='k')
plt.plot(y,x2,  color='gold', linestyle='--', marker='o', markersize=7, linewidth=0.9, markerfacecolor='gold',markeredgewidth=1, markeredgecolor='k')
plt.plot(y,x3,  color='darkorange', linestyle='--', marker='o', markersize=7, linewidth=0.9, markerfacecolor='darkorange',markeredgewidth=1, markeredgecolor='k')
plt.plot(y,x4,  color='red', linestyle='--', marker='o', markersize=7, linewidth=0.9, markerfacecolor='red',markeredgewidth=1, markeredgecolor='k')
plt.plot(y,x5,  color='violet', linestyle='--', marker='o', markersize=7, linewidth=0.9, markerfacecolor='violet',markeredgewidth=1, markeredgecolor='k')
ax = plt.gca()
ax.invert_yaxis()
plt.xlabel('N-okolí [prvků]')
plt.ylabel('Pořadí')
plt.yticks([5.,4.,3.,2.,1.],['5.','4.','3.','2.','1.'])
plt.legend(['Eukleidovská','Čebyševova','Minkowskeho','Mahalanobisova','Hammingova'], bbox_to_anchor=(-0.02,1.02,1.04,0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=5)
plt.show()