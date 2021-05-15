import matplotlib.pyplot as plt
from matplotlib.text import get_rotation

w1 = ['teplo', 'teplo', 'teplo', 'teplo', 'teplo', 'teplo', 'teplo', 'teplo', 'bude', 'teplo', 'teplo', 'az', 'Slezsko', 'n_22', 'bude', 'teplo', 'teplo', 'teplo', 'teplo', 'n_20', 'teplo', 'n_30', 'teplo', 'n_24', 'teplo', 'vedro', 'zitra', 'teplo', 'vedro', 'teplo', 'n_20', 'n_24', 'bude', 'vice_1', 'asi_2', 'odpoledne', 'zitra', 'teplo', 'bude', 'teplo']
w2 = ['teplo', 'teplo', 'teplo', 'teplo', 'teplo', 'teplo', 'teplo', 'teplo', 'teplo', 'teplo', 'teplo', 'bude', 'teplo', 'n_22', 'Slezsko', 'n_20', 'teplo', 'n_30', 'bude', 'zitra', 'teplo', 'teplo', 'teplo', 'teplo', 'bude', 'teplo', 'teplo', 'vedro', 'teplo', 'az', 'odpoledne', 'modry', 'odpoledne', 'zitra', 'teplo', 'teplo', 'teplo', 'teplo', 'zitra', 'n_4']
w3 = ['teplo', 'teplo', 'teplo', 'teplo', 'teplo', 'zacatek', 'zacatek', 'stin_2', 'teplo', 'zacatek', 'teplo', 'teplo', 'pul_1', 'v_tom_2', 'morava', 'nekde', 'teplo', 'plocha', 'teplo', 'bude', 'teplo', 'nizina_1', 'teplo', 'teplo', 'zacatek', 'zacatek', 'stin_2', 'pul_1', 'slunce_3', 'pul_1', 'zacatek', 'obcas', 'pul_1', 'n_11', 'az', 'prvni', 'morava', 'dopoledne', 'teplo', 'teplo']
w4 = ['teplo', 'teplo', 'teplo', 'teplo', 'vedro', 'n_0', 'bude', 'teplota_1', 'teplo', 'odpoledne', 'bude', 'bude', 'teplota_1', 'nizina_1', 'teplota_1', 'teplo', 'bude', 'bude', 'teplota_1', 'teplota_1', 'vedro', 'n_24', 'teplota_1', 'n_23', 'n_31', 'vedro', 'teplo', 'bude', 'slezsko', 'n_28', 'odpoledne', 'bude', 'bude', 'bude', 'teplota_1', 'n_27', 'bude', 'nizina_1', 'n_35', 'n_24']
w5 = ['teplo', 'teplo', 'teplo', 'teplo', 'teplota_1', 'odpoledne', 'bude', 'nizina_1', 'teplo', 'teplota_1', 'bude', 'n_0', 'bude', 'vedro', 'teplota_1', 'teplota_1', 'bude', 'teplota_1', 'teplo', 'bude', 'slezsko', 'teplota_1', 'odpoledne', 'bude', 'vedro', 'n_23', 'teplota_1', 'teplo', 'nizina_1', 'n_27', 'n_24', 'n_35', 'n_28', 'bude', 'n_24', 'vedro', 'bude', 'n_31', 'bude', 'bude']
w6 = ['teplo', 'teplo', 'teplo', 'teplo', 'bude', 'odpoledne', 'teplota_1', 'teplota_1', 'n_0', 'vedro', 'nizina_1', 'bude', 'teplo', 'teplo', 'bude', 'teplota_1', 'bude', 'bude', 'teplota_1', 'teplota_1', 'teplota_1', 'n_24', 'slezsko', 'n_23', 'vedro', 'vedro', 'odpoledne', 'n_28', 'n_31', 'bude', 'teplota_1', 'bude', 'n_27', 'nizina_1', 'n_35', 'bude', 'bude', 'teplo', 'bude', 'teplota_1']
w7 = ['teplo', 'teplo', 'teplo', 'teplo', 'vedro', 'bude', 'n_0', 'nizina_1', 'bude', 'teplota_1', 'n_31', 'teplo', 'bude', 'teplota_1', 'teplo', 'odpoledne', 'vedro', 'teplota_1', 'teplota_1', 'bude', 'bude', 'bude', 'slezsko', 'teplota_1', 'teplota_1', 'bude', 'odpoledne', 'n_35', 'n_24', 'vedro', 'bude', 'n_27', 'teplo', 'teplo', 'teplota_1', 'bude', 'n_23', 'n_5', 'nizina_1', 'teplota_1']
w8 = ['teplo', 'teplo', 'teplo', 'teplo', 'vedro', 'n_0', 'teplota_1', 'teplo', 'teplo', 'bude', 'bude', 'bude', 'vedro', 'teplota_1', 'n_31', 'bude', 'nizina_1', 'odpoledne', 'teplo', 'bude', 'teplota_1', 'n_24', 'teplota_1', 'bude', 'n_23', 'teplota_1', 'bude', 'bude', 'vedro', 'bude', 'n_28', 'bude', 'teplo', 'teplota_1', 'bude', 'bude', 'n_5', 'bude', 'bude', 'bude']



x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]
y1 = [0]*40
y2 = [1]*40
y3 = [2]*40
y4 = [3]*40
y5 = [4]*40
y6 = [5]*40
y7 = [6]*40
y8 = [7]*40

plt.style.use('seaborn')
plt.figure()
plt.plot()
ax = plt.gca()
ax.invert_yaxis()
plt.plot(x,y1,color = 'green', marker='o', markersize=3, linewidth=0.9, markerfacecolor='green',markeredgewidth=1, markeredgecolor='k')
plt.plot(x,y2,color = 'gold', marker='o', markersize=3, linewidth=0.9, markerfacecolor='gold',markeredgewidth=1, markeredgecolor='k')
plt.plot(x,y3,color = 'darkorange', marker='o', markersize=3, linewidth=0.9, markerfacecolor='darkorange',markeredgewidth=1, markeredgecolor='k')
plt.plot(x,y4,color = 'red', marker='o', markersize=3, linewidth=0.9, markerfacecolor='red',markeredgewidth=1, markeredgecolor='k')
plt.plot(x,y5,color = 'violet', marker='o', markersize=3, linewidth=0.9, markerfacecolor='violet',markeredgewidth=1, markeredgecolor='k')
plt.plot(x,y6,color = 'blue', marker='o', markersize=3, linewidth=0.9, markerfacecolor='blue',markeredgewidth=1, markeredgecolor='k')
plt.plot(x,y7,color = 'darkblue', marker='o', markersize=3, linewidth=0.9, markerfacecolor='darkblue',markeredgewidth=1, markeredgecolor='k')
plt.plot(x,y8,color = 'k', marker='o', markersize=3, linewidth=0.9, markerfacecolor='k',markeredgewidth=1, markeredgecolor='k')

for index,data in enumerate(x):
    if w1[index] == 'teplo':
        plt.text(x=index, y=y1[0]-0.08, s="{}".format(w1[index]),fontdict=dict(fontsize=9),  rotation=55, color='darkgreen')
    elif w1[index] != 'teplo':
        plt.text(x=index, y=y1[0]-0.08, s="{}".format(w1[index]),fontdict=dict(fontsize=9),  rotation=55, color='red')
    if w2[index] == 'teplo':
        plt.text(x=index, y=y2[0]-0.08, s="{}".format(w2[index]),fontdict=dict(fontsize=9),  rotation=55, color='darkgreen')
    elif w2[index] != 'teplo':
        plt.text(x=index, y=y2[0]-0.08, s="{}".format(w2[index]),fontdict=dict(fontsize=9),  rotation=55, color='red')
    if w3[index] == 'teplo':
        plt.text(x=index, y=y3[0]-0.08, s="{}".format(w3[index]),fontdict=dict(fontsize=9),  rotation=55, color='darkgreen')
    elif w3[index] != 'teplo':
        plt.text(x=index, y=y3[0]-0.08, s="{}".format(w3[index]),fontdict=dict(fontsize=9),  rotation=55, color='red')
    if w4[index] == 'teplo':
        plt.text(x=index, y=y4[0]-0.08, s="{}".format(w4[index]),fontdict=dict(fontsize=9),  rotation=55, color='darkgreen')
    elif w4[index] != 'teplo':
        plt.text(x=index, y=y4[0]-0.08, s="{}".format(w4[index]),fontdict=dict(fontsize=9),  rotation=55, color='red')
    if w5[index] == 'teplo':
        plt.text(x=index, y=y5[0]-0.08, s="{}".format(w5[index]),fontdict=dict(fontsize=9),  rotation=55, color='darkgreen')
    elif w5[index] != 'teplo':
        plt.text(x=index, y=y5[0]-0.08, s="{}".format(w5[index]),fontdict=dict(fontsize=9),  rotation=55, color='red')
    if w6[index] == 'teplo':
        plt.text(x=index, y=y6[0]-0.08, s="{}".format(w6[index]),fontdict=dict(fontsize=9),  rotation=55, color='darkgreen')
    elif w6[index] != 'teplo':
        plt.text(x=index, y=y6[0]-0.08, s="{}".format(w6[index]),fontdict=dict(fontsize=9),  rotation=55, color='red')
    if w7[index] == 'teplo':
        plt.text(x=index, y=y7[0]-0.08, s="{}".format(w7[index]),fontdict=dict(fontsize=9),  rotation=55, color='darkgreen')
    elif w7[index] != 'teplo':
        plt.text(x=index, y=y7[0]-0.08, s="{}".format(w7[index]),fontdict=dict(fontsize=9),  rotation=55, color='red')
    if w8[index] == 'teplo':
        plt.text(x=index, y=y8[0]-0.08, s="{}".format(w8[index]),fontdict=dict(fontsize=9),  rotation=55, color='darkgreen')
    elif w8[index] != 'teplo':
        plt.text(x=index, y=y8[0]-0.08, s="{}".format(w8[index]),fontdict=dict(fontsize=9),  rotation=55, color='red')
plt.yticks([0,1,2,3,4,5,6,7],['DTW','SoftDTW','Pearson. k.k.','Eukleidovská','Čebyševova','Minkowskeho','Mahalanobisova','Hammingova'],size=11)
plt.xlabel('Index blízkého znaku', size=11)
plt.show()