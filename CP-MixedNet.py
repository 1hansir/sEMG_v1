# from tkinter import _Padding
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from tensorboardX import SummaryWriter
import torchvision
import numpy as np
import torch.utils.data.dataloader as DataLoader
import DataLoader as DL
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, eps=1e-3, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class EMG_NET_1(nn.Module):  # Firstly attention_project different channel
    def __init__(self,channel_p=21, channel_temp=25, conv_pers_window=11, pool_pers_window=3,
                                        time_interval=4050):
        super(EMG_NET_1, self).__init__()

        self.channel_p = channel_p
        self.channel_temp = channel_temp
        self.conv_pers_window = conv_pers_window
        self.pool_pers_window = pool_pers_window
        self.time_interval = time_interval
        # The input size is (trails * channels * 1 * time_indexs)
        # ***CP-Spatio-Temporal Block***
        # Channel Projection
        self.channelProj = nn.Conv2d(7, self.channel_p, 1, stride=1, bias=False)  # (7*1*index)->(21*1*index)
        self.batchnorm_proj_tranf = nn.BatchNorm2d(self.channel_p)
        # Shape Transformation
        self.shapeTrans = nn.Conv2d(self.channel_p, self.channel_p, 1, stride=1,
                                    bias=False)  # (21*1*2700)->(21*1*2700)  这个卷积有什么必要？
        # Temporal Convolution

        self.drop1 = nn.Dropout2d(p=0.5)
        self.conv1 = nn.Conv2d(1, self.channel_temp, (1, self.conv_pers_window), stride=1,
                               bias=False)  # (1*21*2700)->(25*21*2690)   #TIME-feature extraction
        self.batchnorm1 = nn.BatchNorm2d(self.channel_temp, False)
        # Spatial Convolution
        self.drop2 = nn.Dropout2d(p=0.5)
        self.conv2 = nn.Conv2d(self.channel_temp, self.channel_temp, (self.channel_p + 1, 1), stride=1, padding=(1,0),
                               bias=False)  # (25*21*2690)->(25*2*2690)   #spatial-feature extraction
        self.batchnorm2 = nn.BatchNorm2d(self.channel_temp, False)
        # Max Pooling
        self.maxPool1 = nn.MaxPool2d((1, self.pool_pers_window), stride=(1,self.pool_pers_window),
                                     padding=0)  # (25*2*2690)->(25*2*893)

        # ***MS-Conv Block***
        # unDilated Convolution
        self.drop3 = nn.Dropout2d(p=0.5)
        # self.conv3 = nn.Conv2d(25, 50, 1, stride=1, bias=False)        # (25*2*893)->(25*1*893)
        # self.batchnorm3 = nn.BatchNorm2d(50)
        # self.drop4 = nn.Dropout2d(p=0.5)
        self.conv3 = nn.Conv2d(self.channel_temp, self.channel_temp, (1, self.conv_pers_window), stride=1,
                               padding=(0, (self.conv_pers_window - 1) // 2), bias=False)  # (25*2*893)->(25*2*893)
        self.batchnorm3 = nn.BatchNorm2d(25)
        # Dilated Convolution
        self.dropDil = nn.Dropout2d(p=0.5)
        self.dilatedconv = nn.Conv2d(self.channel_temp, self.channel_temp, (1, self.conv_pers_window), stride=1,
                                     padding=(0, self.conv_pers_window - 1), dilation=2,
                                     bias=False)  # (25*2*893)->(25*2*893)
        self.batchnormDil = nn.BatchNorm2d(self.channel_temp)
        # Max pooling after Concatenating
        self.batchnorm_cancat = nn.BatchNorm2d(3 * self.channel_temp)
        self.poolConcatenated = nn.MaxPool2d((1, self.pool_pers_window), stride=(1,self.pool_pers_window),
                                             padding=0)  # (75*1*893)->(75*1*267)

        # ***Classification Block***
        self.drop5 = nn.Dropout(p=0.5)
        self.conv5 = nn.Conv2d(3 * self.channel_temp, 3 * self.channel_temp, (1, self.conv_pers_window),
                               stride=1)  # (75*2*267)->(75*2*257)
        self.batchnorm5 = nn.BatchNorm2d(3 * self.channel_temp)
        self.maxPool2 = nn.MaxPool2d((1, self.pool_pers_window), stride=(1,self.pool_pers_window),
                                     padding=0)  # (75*2*257)->(75*2*86)
        self.fc_dim = (((self.time_interval - (self.conv_pers_window - 1)) // self.pool_pers_window)//self.pool_pers_window - (
                    self.conv_pers_window - 1)) // self.pool_pers_window

        self.conv6 = nn.Conv2d(75,1,kernel_size=(1,140),stride=1)
        # 方案二，不使用全连接层，最后使用一个卷积

        # self.fc = nn.Linear(3 * self.channel_temp * self.fc_dim, 7,bias=False)  # (1*6450)->(1*7)  注意此处的7指的是自由度的7，而最初始channel的7是贴片的7
        # self.softmax = nn.Softmax(dim=1)
        # self.batchnorm6 = nn.BatchNorm1d(7)
        # self.softmax = nn.Softmax(dim=1)       #这个维度貌似不太对，或许可以直接用-1？

        # weight initialization （可以进行记录：conv的 kernel全部初始化为正态分布，batchnorm：weight = 1，bias = 0）
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # print("input size:",x.shape)
        x = F.elu(self.batchnorm_proj_tranf(self.channelProj(x)))
        # print('Channel Projection:',x.shape)
        x = F.elu(self.batchnorm_proj_tranf(self.shapeTrans(x)))
        # print('before Shape Transformation:',x.shape)
        x = torch.transpose(x, 1, 2)  # 交换轴
        # print('after Shape Transformation:',x.shape)
        x = F.elu(self.batchnorm1(self.conv1(self.drop1(x))))
        # print('Temporal convolution:',x.shape)
        x = F.elu(self.batchnorm2(self.conv2(self.drop2(x))))
        # print('Spatial convolution:',x.shape)
        x = self.maxPool1(x)
        # print('Max pooling：',x.shape)

        # x1 = F.elu(self.batchnorm3(self.conv3(self.drop3(x))))
        x_dilated = F.elu(self.batchnormDil(self.dilatedconv(self.dropDil(x))))
        # print('Dilated Convolution1:', x_dilated.shape)
        x_undilated = F.elu(self.batchnorm3(self.conv3(self.drop3(x))))
        # print('Undilated Convolution2:', x_undilated.shape)

        x = torch.cat((x, x_dilated, x_undilated), dim=1)
        # print('Concatenated:', x.shape)

        x = self.poolConcatenated(self.batchnorm_cancat(x))
        # print('MixedScaleConv:', x.shape)

        x = F.elu(self.batchnorm5(self.conv5(self.drop5(x))))
        # print('Conv5:', x.shape)
        x = self.maxPool2(x)
        # print('maxPool2:', x.shape)
        # x = x.view(-1, 3*self.channel_temp*self.fc_dim)

        x = F.relu(self.conv6(x))
        # print('conv6:', x.shape)
        # x = self.softmax(x)
        # print('softmax:', x.shape)
        x = torch.squeeze(x)
        # print("x_squeeze",x.shape)

        # 获得X中的最大值,防止softmax函数上溢出，但是好像不是这个问题，这个应该是loss函数产生的下溢出
        val_max,_ = torch.max(x,dim=1)
        val_max = torch.cat((val_max[:, np.newaxis, :],val_max[:, np.newaxis, :]),dim = 1)
        # print(val_max ,val_max.shape)

        x = F.softmax(x-val_max, dim=1)
        # print("softmax:",x.shape)
        # print(x)

        return x  # 模型结尾使用了softmax函数，因此损失函数使用NLLloss()，softmax应该作用于2的维度


def train(model, device, train_loader, optimizer, epoch,
          log_interval=100, ):  # 每过100个batch输出一次观察，这样至少需要12800个数据，但并不存在如此多数据，因此一般只有在batch_idx=0时才会输出一次观察
    model.train()
    correct = 0
    loss_fn = torch.nn.MultiLabelSoftMarginLoss(reduction='mean')

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        # print("target: ",target)
        # print("output: ",output)

        loss = loss_fn(output[:,0,:], target)
        # - loss_fn(output[:,1,:], target)
        # loss = F.nll_loss(output, target.squeeze())  #target 的 shape多了1维

        # loss_fun = nn.CrossEntropyLoss()
        # loss = loss_fun(output, target)
        loss.backward()
        optimizer.step()
        '''
        for i,(name, param) in enumerate(model.named_parameters()):
            if 'bn' not in name:
                writer.add_histogram(name, param, 0)
                writer.add_scalar('loss', loss, i)
                loss = loss * 0.5
        '''
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:0f}%)]\tLoss:{:.6f}".format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()
            ))
            '''
            writer.add_scalar(
                "Training loss",
                loss.item(),
                epoch * len(train_loader)
            )
            '''

    print("Trainning accuracy:", 100. * correct / (len(train_loader.dataset)*batch_size))


def val(model, device, val_loader, optimizer):
    model.train()
    val_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(val_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        # loss = F.nll_loss(output, target.squeeze())

        loss_fun = nn.CrossEntropyLoss()
        loss = loss_fun(output, target)
        loss.backward()
        optimizer.step()

        val_loss += loss * batch_size
        pred = output.argmax(dim=1, keepdim=True)
        # correct = output.eq(target.view_as(pred)).sum().item()
        correct += pred.eq(target.view_as(
            pred)).sum().item()  # view_as reshape the [target] and eq().sum().item()  get the sum of the correct validation
        if batch_idx == 0:
            print("pred:", output[0])
            print("true:", target[0])
    val_loss /= len(val_loader.dataset)
    print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # loss_fun = nn.CrossEntropyLoss()
            # test_loss += loss_fun(output, target) * batch_size

            test_loss += F.nll_loss(output, target.squeeze(), reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def create_summary_writer(model, data_loader, log_dir):
    writer = SummaryWriter(logdir=log_dir)
    data_loader_iter = iter(data_loader)
    x, y = next(data_loader_iter)
    try:
        writer.add_graph(model, x)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))


if __name__ == "__main__":

    # Configs and Hyperparameters
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Pytorch Version:", torch.__version__)
    print('device={}'.format(device))
    batch_size = 16
    val_batch_size = 32
    learning_rate = 1e-7
    weight_decay = 0.01

    train_dataset = DL.import_sEMGData_train()
    train_loader = DataLoader.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = EMG_NET_1().to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    epoches = 500
    for epoch in range(1, epoches + 1):
        train(model, device, train_loader, optimizer, epoch)
        # val(model, device, val_loader, optimizer)
        # test(model, device, test_loader)
    save_model = False
    if (save_model):
        torch.save(model.state_dict(), "weights.pt")

    '''
    for sub in range(1,9):
        #train_dataset = DL.import_EEGData(0, 1, 'data/A0')
        X = np.load("processed_data/train_data_BCI_{}.npy".format(sub))
        y = np.load("processed_data/train_label_BCI_{}.npy".format(sub))
        y = y[:, np.newaxis]
        print(X.shape, y.shape)
        train_dataset = DL.EEGDataset(X, y)

        X = np.load("processed_data/val_data_BCI_{}.npy".format(sub))
        y = np.load("processed_data/val_label_BCI_{}.npy".format(sub))
        y = y[:, np.newaxis]
        print("shape of val set:", X.shape, y.shape)
        val_dataset = DL.EEGDataset(X, y)

        X = np.load("processed_data/test_data_BCI_{}.npy".format(sub))
        y = np.load("processed_data/test_label_BCI_{}.npy".format(sub))
        y = y[:, np.newaxis]
        print("shape of test set:", X.shape, y.shape)
        test_dataset = DL.EEGDataset(X, y)


        train_loader = DataLoader.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
        test_loader = DataLoader.DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False)

        model = EMG_NET_1().to(device)
        #writer = SummaryWriter('tensorboard_logs')



        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        epoches = 500
        for epoch in range(1, epoches + 1):
            train(model, device, train_loader, optimizer, epoch)
            val(model, device, val_loader, optimizer)
            test(model, device, test_loader)
        save_model = False
        if (save_model):
            torch.save(model.state_dict(), "weights.pt")

        print("-----------------------")
        print("The training of {}th subject is complete!".format(sub))
        print("-----------------------")
        '''

    print("test training is done")
    print("\nAll train have been done！")
