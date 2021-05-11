import matplotlib.pyplot as plt
from matplotlib.text import get_rotation

plt.style.use('seaborn')
top_counts = [1,1,1,3,3,3,5,5,5,10,10,10,20,20,20,30,30,30]
graph_data_1 = [64.70801397599057, 60.60175076688456, 55.958299253700595, 49.510481408223875, 39.843454364754535, 32.6477387950214]
graph_data_2 = [63.762517721965885, 57.91937169111082, 54.6972599585476, 47.80045738820823, 38.76572393604166, 31.8707235351048]
graph_data = []
x = ['1:d','1:u','3:d','3:u','5:d','5:u','10:d','10:u','20:d','20:u','30:d','30:u']
for j in range(len(graph_data_1)):
    graph_data.append(graph_data_1[j])
    graph_data.append(graph_data_2[j])
graph_data1 = [100,100,100,100,100,100]*2

plt.figure()
plt.grid(True)
plt.plot(0,0, markersize=0, linewidth=0)
plt.bar(x,graph_data1, color='red', alpha=1, width=0.4, edgecolor='black', linewidth=1)
plt.bar(x,graph_data, color='green', alpha=1, width=0.4, edgecolor='black', linewidth=1)
plt.xlabel('Počet nejbližších projevů [znak]')
plt.ylabel('Zastoupení projevu se stejným významem [%]')
for index,data in enumerate(graph_data):
    plt.text(x=index-0.20, y=data+1, s="{:.2f} %".format(data) , fontweight='bold',fontdict=dict(fontsize=10),  rotation=-90, color='w')
plt.legend(['','Rozdíl významu', 'Shoda významu'], bbox_to_anchor=(-0.1,1.02,0.6,0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=3)
plt.show()