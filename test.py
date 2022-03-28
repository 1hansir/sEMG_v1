import matplotlib.pyplot as plt
import numpy as np
import matplotlib

data = np.genfromtxt("EMG.txt",dtype=int)  # 将文件中数据加载到data数组里

len = 4000
start = 138000
data = data[start:start+len]
x = np.linspace(0,len,len)
print(data.shape[1])
data_cut = []

for i in range(data.shape[0]):
    data_cut.append( data[i][2:data.shape[1]] - data[i][1])

data_cut = np.array(data_cut)
plt.plot(x,data_cut)
plt.show()

print(data_cut)
