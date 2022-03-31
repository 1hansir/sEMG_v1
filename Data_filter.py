from tkinter.tix import MAIN
import numpy as np
import matplotlib.pyplot as plt
from pip import main
from scipy import signal
from sklearn.decomposition import FastICA
import pywt

biggest_channel = 7
dir = 'EMG.txt'
FPS = 270 
time_interval = 15

#139000  10000 是一组好数据
#1490000  10000 是一组工频


# print(data)
# print(data.shape)   # trial * time_index * channel
# data = np.swapaxes(data,0,1)   # 此处将channel数转置到dim=0,数据形状为7*n

#butterworth 滤波器
def Filter(low = 0.5,high = 100,band = (48,52),fs=270):
    sos_1 = signal.butter(5, low, btype='highpass', analog=False, output='sos', fs=fs)
    sos_2 = signal.butter(15,high,btype='lowpass',analog=False,output='sos',fs=fs)
    sos_3 = signal.butter(5,band,btype='bandstop',analog=False,output='sos',fs=fs)
    filtered = signal.sosfilt(sos_1, data)  # 滤波
    filtered = signal.sosfilt(sos_2,filtered)
    filtered = signal.sosfilt(sos_3,filtered)

    return filtered
#---------

# ICA 独立主成分分析
def ICA(data_,n_component =2):
    ica = FastICA(n_components=n_component)

    u = ica.fit_transform(data_)

    u = u.T

    return u

#--------

# STFT 
def SFTF(data,fs=270,window='hann',nperseg=120,cut_interval=2000):
    f, t, Zxx = signal.stft(data[cut_interval:],fs = fs,window =window,
                                nperseg =nperseg )     #注意滤波之后的时域信号在最开始会有奇怪的畸变，所以在短时变换前截取前面一小段，能够使结果更客观
    # 窗口的长度会影响时间分辨率和频率分辨率。窗口越长，截取的信号越长，时间分辨率越低，频率分辨率越高；窗口越短，截取的信号越短，时间分辨率越高，频率分辨率越低。

    # plt.plot(x,data)
    # plt.plot(x,data_i)
    plt.pcolormesh(t, f, np.abs(Zxx))
    plt.colorbar()
    plt.tight_layout()
    # plt.plot(filtered)
    #plt.ylim(0,250)

    return f,t,Zxx

#---------


# DWT
def DWT(filtered,wave = 'sym3',wave_mode = 'constant'):
    w = pywt.Wavelet(wave)
    cA, cD = pywt.dwt(filtered, wavelet=w, mode=wave_mode)
    print(cA, cD)   # CA 是粗略信息、CD 是细节信息
    filtered_wave =  pywt.idwt(cA, None, wave)


    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    ax1.plot(x,data)
    ax2.plot(x,filtered_wave)

    return filtered_wave

#-------



'''
ax1 = plt.subplot(511)

ax2 = plt.subplot(512)

ax3 = plt.subplot(513)

ax4 = plt.subplot(514)

ax5 = plt.subplot(515)

ax1.plot(x,u[0,:])

ax2.plot(x,u[1,:])

ax3.plot(x,data)

ax4.plot(x,filtered)

ax5.plot(Zxx[0,:],Zxx[])

'''
if __name__=='__main__':

    len_window = 10000  # 定义一个trial的时间长度
    max_alt = 12060687   # 最高点-bias
    data_load = np.genfromtxt(dir , dtype=int)  # 将文件中数据以int类型加载到data数组里
    start_point = 138000
    start_point_2 = 149000

    data = []
    index = []

    x = np.linspace(0,len_window,len_window)

    data = data_load[start_point:start_point+len_window,2]-data_load[start_point:start_point+len_window,1]
    data_f = data_load[start_point_2:start_point_2+len_window,2]-data_load[start_point_2:start_point_2+len_window,1]
    # print(data)

    data = np.array(data)
    data_f = np.array(data_f)
    data_i = np.concatenate((data.reshape(-1,1),data_f.reshape(-1,1)),axis=1)

    
    plt.show()

# 剔除坏值(凡是有大于量程数据的trial全部剔除) 
# 















''' 
delete_list = []
for i in range(len(data)):
    if np.any(data[i] >= max_alt):
        delete_list.append(i)
        print("This trial =>{} has overflowed values".format(i))

data = np.delete(data, delete_list, 0)
#index = np.delete(index, delete_list, 0)
data = data[:,:,np.newaxis,:]
data = np.swapaxes(data,1,3)

index = np.array(index)
print("data_size:",data.shape)
print("index_size:", index.shape)
'''