import matplotlib.pyplot as plt
from matplotlib.text import get_rotation
"""
w1 = ['teplo #1', 'teplo #9', 'teplo #19', 'teplo #23', 'teplo #26', 'teplo #43', 'teplo #17', 'teplo #14', 'bude', 'teplo #24', 'teplo #35', 'az', 'Slezsko', 'n_22', 'bude', 'teplo #39', 'teplo #25', 'teplo #41', 'teplo #27', 'n_20', 'teplo #22', 'n_30', 'teplo #12', 'n_24', 'teplo #15', 'vedro', 'zitra', 'teplo #28', 'vedro', 'teplo #2', 'n_20', 'n_24', 'bude', 'vice_1', 'asi_2', 'odpoledne', 'zitra', 'teplo #30', 'bude', 'teplo #42']
w2 = ['teplo #1', 'teplo #9', 'teplo #19', 'teplo #23', 'teplo #26', 'teplo #43', 'teplo #17', 'teplo #14', 'teplo #24', 'teplo #39', 'teplo #35', 'bude', 'teplo #22', 'n_22', 'Slezsko', 'n_20', 'teplo #25', 'n_30', 'bude', 'zitra', 'teplo #41', 'teplo #27', 'teplo #30', 'teplo #28', 'bude', 'teplo #12', 'teplo #15', 'vedro', 'teplo #2', 'az', 'odpoledne', 'modry', 'odpoledne', 'zitra', 'teplo #13', 'teplo #42', 'teplo #40', 'teplo #33', 'zitra', 'n_4']
w3 = ['teplo #1', 'teplo #9', 'teplo #26', 'teplo #34', 'teplo #23', 'zacatek', 'zacatek', 'stin_2', 'teplo #19', 'zacatek', 'teplo #40', 'teplo #17', 'pul_1', 'v_tom_2', 'morava', 'nekde', 'teplo #14', 'plocha', 'teplo #24', 'bude', 'teplo #43', 'nizina_1', 'teplo #2', 'teplo #41', 'zacatek', 'zacatek', 'stin_2', 'pul_1', 'slunce_3', 'pul_1', 'zacatek', 'obcas', 'pul_1', 'n_11', 'az', 'prvni', 'morava', 'dopoledne', 'teplo #12', 'teplo #25']
w4 = ['teplo #1', 'teplo #9', 'teplo #23', 'teplo #26', 'vedro', 'n_0', 'bude', 'teplota_1', 'teplo #14', 'odpoledne', 'bude', 'bude', 'teplota_1', 'nizina_1', 'teplota_1', 'teplo #19', 'bude', 'bude', 'teplota_1', 'teplota_1', 'vedro', 'n_24', 'teplota_1', 'n_23', 'n_31', 'vedro', 'teplo #17', 'bude', 'slezsko', 'n_28', 'odpoledne', 'bude', 'bude', 'bude', 'teplota_1', 'n_27', 'bude', 'nizina_1', 'n_35', 'n_24']
w5 = ['teplo #1', 'teplo #9', 'teplo #23', 'teplo #26', 'teplota_1', 'odpoledne', 'bude', 'nizina_1', 'teplo #19', 'teplota_1', 'bude', 'n_0', 'bude', 'vedro', 'teplota_1', 'teplota_1', 'bude', 'teplota_1', 'teplo #14', 'bude', 'slezsko', 'teplota_1', 'odpoledne', 'bude', 'vedro', 'n_23', 'teplota_1', 'teplo #37', 'nizina_1', 'n_27', 'n_24', 'n_35', 'n_28', 'bude', 'n_24', 'vedro', 'bude', 'n_31', 'bude', 'bude']
w6 = ['teplo #1', 'teplo #9', 'teplo #23', 'teplo #26', 'bude', 'odpoledne', 'teplota_1', 'teplota_1', 'n_0', 'vedro', 'nizina_1', 'bude', 'teplo #14', 'teplo #19', 'bude', 'teplota_1', 'bude', 'bude', 'teplota_1', 'teplota_1', 'teplota_1', 'n_24', 'slezsko', 'n_23', 'vedro', 'vedro', 'odpoledne', 'n_28', 'n_31', 'bude', 'teplota_1', 'bude', 'n_27', 'nizina_1', 'n_35', 'bude', 'bude', 'teplo #17', 'bude', 'teplota_1']
w7 = ['teplo #1', 'teplo #9', 'teplo #23', 'teplo #26', 'n_0', 'vedro', 'nizina_1', 'bude', 'teplota_1', 'bude', 'teplo #14', 'teplota_1', 'odpoledne', 'teplo #19', 'teplota_1', 'bude', 'bude', 'n_31', 'teplota_1', 'n_28', 'teplota_1', 'bude', 'vedro', 'teplota_1', 'bude', 'n_23', 'bude', 'bude', 'vedro', 'slezsko', 'odpoledne', 'n_24', 'vice_1', 'teplo #17', 'bude', 'bude', 'teplo #22', 'vedro', 'nizina_1', 'n_27']
w8 = ['teplo #1', 'teplo #9', 'teplo #23', 'teplo #26', 'vedro', 'n_0', 'teplota_1', 'teplo #14', 'teplo #17', 'bude', 'bude', 'bude', 'vedro', 'teplota_1', 'n_31', 'bude', 'nizina_1', 'odpoledne', 'teplo #19', 'bude', 'teplota_1', 'n_24', 'teplota_1', 'bude', 'n_23', 'teplota_1', 'bude', 'bude', 'vedro', 'bude', 'n_28', 'bude', 'teplo #22', 'teplota_1', 'bude', 'bude', 'n_5', 'bude', 'bude', 'bude']
"""

w1 = ['zitra #1', 'zitra #5', 'zitra #7', 'zitra #28', 'zitra #16', 'zitra #27', 'zitra #14', 'zitra #19', 'zitra #24', 'zitra #29', 'zitra #30', 'zitra #21', 'zitra #25', 'zitra #20', 'zitra #8', 'zitra #12', 'zitra #31', 'zitra #15', 'zitra #13', 'zitra #23', 'zitra #4', 'zitra #9', 'bude', 'bude', 'zitra #17', 'teplo', 'bude', 'zitra #11', 'bude', 'bude', 'bude', 'cechy_1', 'cechy_1', 'zitra #3', 'teplo', 'modry', 'bude', 'bude', 'zitra #6', 'bude']
w2 = ['zitra #1', 'zitra #5', 'zitra #7', 'zitra #28', 'zitra #27', 'zitra #16', 'zitra #19', 'zitra #21', 'zitra #14', 'zitra #25', 'zitra #30', 'zitra #29', 'zitra #24', 'zitra #20', 'zitra #12', 'zitra #8', 'zitra #31', 'zitra #15', 'zitra #13', 'zitra #23', 'zitra #4', 'zitra #17', 'bude', 'bude', 'teplo', 'zitra #9', 'bude', 'bude', 'cechy_1', 'cechy_1', 'teplo', 'bude', 'bude', 'bude', 'modry', 'bude', 'cechy_1', 'bude', 'zitra #11', 'bude']
w3 = ['zitra #1', 'zitra #19', 'zitra #7', 'zitra #16', 'zitra #14', 'teplo', 'buce', 'bude', 'den', 'den', 'zitra #29', 'zitra #5', 'zitra #28', 'bude', 'slunce_1', 'od_1', 'az', 'akorat', 'bude', 'zitra #17', 'n_3', 'az', 'metr_1', 'az', 'zitra #15', 'slunce_1', 'bude', 'i_1', 'take', 'az', 'bude', 'take', 'az', 'take', 'bude', 'take', 'take', 'bude', 'az', 'bude']
w4 = ['zitra #1', 'zitra #16', 'zitra #7', 'patek', 'zitra #14', 'kolem', 'n_20', 'zitra #5', 'zitra #28', 'teplo', 'zitra #29', 'nekde', 'kolem', 'zitra #23', 'zitra #30', 'změna', 'jeste', 'vedro', 'nekde', 'patek', 'teplo', 'jz', 'nekde', 'nekdy', 'bude', 'zitra #19', 'zitra #17', 'zitra #24', 'zitra #15', 'bude', 'asi_2', 'teplo', 'kolem', 'jih', 'hory_2', 'den', 'slaby_2', 'zitra #8', 'n_3', 'teplo']
w5 = ['zitra #1', 'zitra #16', 'zitra #7', 'zitra #14', 'patek', 'zitra #5', 'změna', 'kolem', 'n_20', 'zitra #28', 'zitra #29', 'teplo', 'patek', 'kolem', 'nekde', 'nekde', 'zitra #17', 'zitra #30', 'asi_2', 'vedro', 'zitra #23', 'jz', 'jeste', 'zitra #19', 'nekdy', 'zitra #24', 'nekde', 'teplo', 'zitra #15', 'jih', 'nekde', 'zitra #26', 'zitra #8', 'hory_2', 'n_3', 'kolem', 'horko_1', 'bude', 'den', 'teplo']
w6 = ['zitra #1', 'zitra #16', 'zitra #7', 'patek', 'zitra #14', 'zitra #5', 'kolem', 'n_20', 'zitra #28', 'teplo', 'zitra #29', 'změna', 'nekde', 'kolem', 'zitra #23', 'nekde', 'zitra #30', 'patek', 'vedro', 'jeste', 'jz', 'zitra #17', 'zitra #19', 'teplo', 'nekdy', 'nekde', 'asi_2', 'bude', 'zitra #24', 'zitra #15', 'jih', 'zitra #26', 'hory_2', 'kolem', 'zitra #8', 'den', 'n_3', 'nekde', 'teplo', 'bude']
w7 = ['zitra #1', 'zitra #16', 'zitra #7', 'patek', 'zitra #14', 'zitra #28', 'n_20', 'teplo', 'kolem', 'zitra #5', 'nekde', 'vedro', 'kolem', 'zitra #23', 'zitra #29', 'nekde', 'jeste', 'zitra #30', 'nekdy', 'teplo', 'změna', 'zitra #19', 'bude', 'patek', 'zitra #17', 'nekde', 'bude', 'slaby_2', 'zitra #15', 'teplo', 'zitra #8', 'zitra #24', 'asi_2', 'jz', 'jih', 'teplo', 'bude', 'hory_2', 'n_-5', 'horko_1']
w8 = ['zitra #1', 'zitra #16', 'zitra #7', 'patek', 'zitra #14', 'n_20', 'zitra #28', 'kolem', 'teplo', 'zitra #29', 'zitra #5', 'zitra #30', 'kolem', 'nekde', 'bude', 'teplo', 'zitra #23', 'jeste', 'nekde', 'vedro', 'nekdy', 'zitra #17', 'patek', 'jz', 'nekde', 'zitra #19', 'slaby_2', 'bude', 'změna', 'teplo', 'kolem', 'zitra #24', 'teplo', 'od_2', 'teplo', 'nekde', 'n_-5', 'zitra #15', 'kolem', 'zitra #8']

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
    
    if ''.join(i for i in w1[index] if i.isalpha()) == 'zitra':
        plt.text(x=index, y=y1[0]-0.08, s="{}".format(w1[index]),fontdict=dict(fontsize=9),  rotation=55, color='darkgreen')
    elif ''.join(i for i in w1[index] if i.isalpha()) != 'zitra':
        plt.text(x=index, y=y1[0]-0.08, s="{}".format(w1[index]),fontdict=dict(fontsize=9),  rotation=55, color='red')
    if''.join(i for i in w2[index] if i.isalpha()) == 'zitra':
        plt.text(x=index, y=y2[0]-0.08, s="{}".format(w2[index]),fontdict=dict(fontsize=9),  rotation=55, color='darkgreen')
    elif ''.join(i for i in w2[index] if i.isalpha()) != 'zitra':
        plt.text(x=index, y=y2[0]-0.08, s="{}".format(w2[index]),fontdict=dict(fontsize=9),  rotation=55, color='red')
    if ''.join(i for i in w3[index] if i.isalpha()) == 'zitra':
        plt.text(x=index, y=y3[0]-0.08, s="{}".format(w3[index]),fontdict=dict(fontsize=9),  rotation=55, color='darkgreen')
    elif ''.join(i for i in w3[index] if i.isalpha()) != 'zitra':
        plt.text(x=index, y=y3[0]-0.08, s="{}".format(w3[index]),fontdict=dict(fontsize=9),  rotation=55, color='red')
    if ''.join(i for i in w4[index] if i.isalpha()) == 'zitra':
        plt.text(x=index, y=y4[0]-0.08, s="{}".format(w4[index]),fontdict=dict(fontsize=9),  rotation=55, color='darkgreen')
    elif ''.join(i for i in w4[index] if i.isalpha()) != 'zitra':
        plt.text(x=index, y=y4[0]-0.08, s="{}".format(w4[index]),fontdict=dict(fontsize=9),  rotation=55, color='red')
    if ''.join(i for i in w5[index] if i.isalpha()) == 'zitra':
        plt.text(x=index, y=y5[0]-0.08, s="{}".format(w5[index]),fontdict=dict(fontsize=9),  rotation=55, color='darkgreen')
    elif ''.join(i for i in w5[index] if i.isalpha()) != 'zitra':
        plt.text(x=index, y=y5[0]-0.08, s="{}".format(w5[index]),fontdict=dict(fontsize=9),  rotation=55, color='red')
    if ''.join(i for i in w6[index] if i.isalpha()) == 'zitra':
        plt.text(x=index, y=y6[0]-0.08, s="{}".format(w6[index]),fontdict=dict(fontsize=9),  rotation=55, color='darkgreen')
    elif ''.join(i for i in w6[index] if i.isalpha()) != 'zitra':
        plt.text(x=index, y=y6[0]-0.08, s="{}".format(w6[index]),fontdict=dict(fontsize=9),  rotation=55, color='red')
    if ''.join(i for i in w7[index] if i.isalpha()) == 'zitra':
        plt.text(x=index, y=y7[0]-0.08, s="{}".format(w7[index]),fontdict=dict(fontsize=9),  rotation=55, color='darkgreen')
    elif ''.join(i for i in w7[index] if i.isalpha()) != 'zitra':
        plt.text(x=index, y=y7[0]-0.08, s="{}".format(w7[index]),fontdict=dict(fontsize=9),  rotation=55, color='red')
    if ''.join(i for i in w8[index] if i.isalpha()) == 'zitra':
        plt.text(x=index, y=y8[0]-0.08, s="{}".format(w8[index]),fontdict=dict(fontsize=9),  rotation=55, color='darkgreen')
    elif ''.join(i for i in w8[index] if i.isalpha()) != 'zitra':
        plt.text(x=index, y=y8[0]-0.08, s="{}".format(w8[index]),fontdict=dict(fontsize=9),  rotation=55, color='red')
plt.yticks([0,1,2,3,4,5,6,7],['DTW','SoftDTW','Pearson. k.k.','Eukleidovská','Čebyševova','Minkowskeho','Mahalanobisova','Hammingova'],size=11)
plt.xlabel('Index blízkého znaku', size=11)
plt.show()