import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import Data_filter as df
from scipy import signal


dir = 'sEMG_datasets/up_abduction/'
data = np.genfromtxt(dir+"EMG.txt",dtype=int)  # 将文件中数据加载到data数组里

len = 200000
start = 13800
fs=1000
data = data[start:start+len]
x = np.linspace(0,len,len)
print(data.shape[1])
data_cut = []
'''
ax1 = plt.subplot(411)
ax2 = plt.subplot(412)
ax3 = plt.subplot(413)
ax4 = plt.subplot(414)
'''

sos_1 = signal.butter(10, 5, btype='highpass', analog=False, output='sos', fs=fs)
sos_2 = signal.butter(15, 200, btype='lowpass', analog=False, output='sos', fs=fs)
sos_3 = signal.butter(10, (49,51), btype='bandstop', analog=False, output='sos', fs=fs)
sos_4 = signal.butter(10, (30,40), btype='bandpass', analog=False, output='sos', fs=fs)

def ax(i):
    num = 409+i
    return plt.subplot(num)

for i in range(2,3):
    data_cut = ( data[:,i] - data[:,1])
    data_cut = data_cut - np.mean(data_cut)   # mention! here must be a normalization processing
    data_cut = np.array(data_cut)

    #print(data_cut.shape)



    data_cut = df.Filter(data_cut,5,200,(48,52),fs)
    # data_cut = signal.sosfilt(sos_4,data_cut)
    data_cut = df.DWT(data_cut)
    df.STFT(data_cut, fs, 'hann', 256, 100, 250)

    # ax(i).plot(x, data_cut)




# data_cut = np.array(data_cut)
# plt.plot(x,data_cut)
plt.show()

print(data_cut)
