import matplotlib.pyplot as plt
from matplotlib.text import get_rotation

plt.style.use('seaborn')
top_counts = [1,1,1,3,3,3,5,5,5,10,10,10,20,20,20,30,30,30]
graph_data_1 = [63.83, 59.39, 56.03, 48.71, 39.46, 32.56]
graph_data_2 = [63.83, 59.0, 55.87, 48.49, 39.27, 32.43]
graph_data_3 = [63.83, 59.0, 55.87, 48.49, 39.27, 32.43]
graph_data = []
x = ['1:l','1:kv','1:k','3:l','3:kv','3:k','5:l','5:kv','5:k','10:l','10:kv','10:k','20:l','20:kv','20:k','30:l','30:kv','30:k']
for j in range(len(graph_data_1)):
    graph_data.append(graph_data_1[j])
    graph_data.append(graph_data_2[j])
    graph_data.append(graph_data_3[j])
graph_data_x = [100,100,100,100,100,100]*3

plt.figure()
plt.grid(True)
plt.plot(0, 0, markersize=0, linewidth=0)
plt.bar(x, graph_data_x, color='red', alpha=1, width=0.4, edgecolor='black', linewidth=1)
plt.bar(x, graph_data, color='green', alpha=1, width=0.4, edgecolor='black', linewidth=1)
plt.xlabel('velikost n-oblasti [projevů] : řád interpolace [l = lineární, kv = kvadratická, k = kubická]')
plt.ylabel('zastoupení projevu se stejným významem [%]')
for index,data in enumerate(graph_data):
    plt.text(x=index-0.28, y=data+1, s="{:.2f} %".format(data) , fontweight='bold',fontdict=dict(fontsize=10),  rotation=-90, color='w')
plt.legend(['','Rozdíl významu', 'Shoda významu'], bbox_to_anchor=(-0.1,1.02,0.6,0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=3)
plt.show()