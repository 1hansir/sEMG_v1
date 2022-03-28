import numpy as np
import torch
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader


class sEMGDataset(Dataset.Dataset):
    def __init__(self, Data, Label):
        self.Data = Data    # Tensor(X*(22*1*1125))
        self.Label = Label  # Tensor(X*1)

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, index):
        data = torch.Tensor(self.Data[index])
        label = torch.LongTensor(self.Label[index]).squeeze()
        return data, label
        # return self.Data[index], self.Label[index]

def import_sEMGData_train(biggest_channel = 7, dir = 'EMG.txt',FPS = 270 ,time_interval = 15):

    # FPS = 270    # 每秒帧数，取决于肌电采集器自身的设定
    # time_interval = 15  # 每一个sample的秒数

    len_window = FPS * time_interval  # 定义一个trial的时间长度
    max_alt = 12060687   # 最高点-bias
    data_load = np.genfromtxt(dir , dtype=int)  # 将文件中数据以int类型加载到data数组里

    data = []
    index = []

    for i in range(data_load.shape[0]//len_window):
        data_cut = data_load[i * len_window: (i+1)*len_window]    # 取数据的一个观察窗

        data_ = []
        for j in range(data_cut.shape[0]):
            data_.append(data_cut[j][2:biggest_channel+2] - data_cut[j][1])  # 这步操作之后，data_cut 的形状是n*9,n代表时间点

        if( i == 0):
            data = np.array(data_)[np.newaxis,:,:]

        else:
            data_ = np.array(data_)[np.newaxis,:,:]
            data = np.concatenate([data,data_],axis=0)

    data = np.array(data)
    # print(data)
    # print(data.shape)   # trial * time_index * channel
    # data = np.swapaxes(data,0,1)   # 此处将channel数转置到dim=0,数据形状为7*n

    # 剔除坏值(凡是有大于量程数据的trial全部剔除)  
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
    dataset = sEMGDataset(data, index)

    return dataset


if __name__ == '__main__':
    import_sEMGData_train(biggest_channel=2, dir='EMG.txt')