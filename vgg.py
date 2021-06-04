import torch
import torch.nn as nn


class Classifier(nn.Module):
    # 搭建网络模型;
    # input 维度 [N, 3, 128, 128]
    # output 维度 [N, 11]

    # TODO: 改为卷积神经网络
    # nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    # nn.MaxPool2d(kernel_size, stride, padding)
    # nn.BatchNorm2d(channels)
    # nn.ReLU()
    # ...

    def __init__(self):
        super(Classifier, self).__init__()
        # super：调用父类(超类)
        self.fc = nn.Sequential(nn.Linear(3 * 128 * 128, 1024),
                                # 用于设置网络中的全连接层
                                # in_features由输入张量的形状决定，out_features则决定了输出张量的形状
                                nn.ReLU(),
                                # 激活函数
                                nn.Linear(1024, 11))
        # nn.Sequential:一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，
        # 同时以神经网络模块为元素的有序字典也可以作为传入参数

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        # x.size()[0]中[0]表示维度
        # x.view(x.size()[0], -1)：将前面操作输出的多维度的tensor展平成一维，然后输入分类器
        # -1是自适应分配，指在不知道函数有多少列的情况下，根据原tensor数据自动分配列数。
        return self.fc(x)
