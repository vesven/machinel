import os
import random
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from model import Classifier


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    #每次返回的卷积算法将是确定的


# 读取数据集文件;
# 文件名格式为label+id，即"1_2.jpg"表示label为1，id为2的jpg图片;
def readfile(path):
    image_dir = sorted(os.listdir(path))
    # 返回path指定的文件夹包含的文件或文件夹的名字的列表、返回重新排序的列表。
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    # dtype:返回的是该数组的数据类型,uint:不带符号
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file_name in enumerate(image_dir):
    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
    # 一般用在 for 循环当中。
        img = cv2.imread(os.path.join(path, file_name))
    # imread：读取图片文件中的数据
    # os.path,join: 把目录和文件名合成一个路径
        x[i, :, :] = cv2.resize(img, (128, 128))
    # 尺度缩放
        y[i] = int(file_name.split("_")[0])
    return x, y


# 设置随机数种子
setup_seed(1896)

# 分別将 training set、validation set 用 readfile 函数读进来;
workspace_dir = './food-3'
print("Reading data")
train_x, train_y = readfile(os.path.join(workspace_dir, "training"))
print("Size of training data = {}".format(len(train_x)))
val_x, val_y = readfile(os.path.join(workspace_dir, "validation"))
print("Size of validation data = {}".format(len(val_x)))

# 训练集图片预处理
# ToTensor: Convert a PIL image(H, W, C) in range [0, 255] to
#           a torch.Tensor(C, H, W) in the range [0.0, 1.0]
# c：图像通道数
# w：图像宽度
# h：图像高度
# transforms.Compose是一个内置函数
train_transform = transforms.Compose([
    transforms.ToPILImage(),
# convert a tensor to PIL image
    transforms.ToTensor(),
])

# 验证集图片预处理
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])


# 构建训练集、验证集;
class ImgDataset(Dataset):
    def __init__(self, x, y, transform):
        self.x = x
        # label is required to be a LongTensor（64位整型）
        self.y = torch.LongTensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        X = self.transform(X)
        Y = self.y[index]
        return X, Y


batch_size = 16
train_set = ImgDataset(train_x, train_y, train_transform)
val_set = ImgDataset(val_x, val_y, val_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
# DataLoader：形成可迭代对象
# batch_size(int, optional): 每个batch有多少个样本
# shuffle(bool, optional): 在每个epoch开始的时候，对数据进行重新排序
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# 检测是否能够使用GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 搭建模型 model;
model = Classifier().to(device)

# 构造损失函数 loss;
loss = nn.CrossEntropyLoss()
# 交叉熵损失函数

learning_rate = 0.001

# 构造优化器 optimizer;
optimizer = torch.optim.Adam(model.parameters(),
                             lr=learning_rate)
# optimizer 优化器使用Adam，初始化

# 设定训练次数 num_epoch;
num_epoch = 10
val_acc_best = 0.0

# 训练 并print每个epoch的结果;
for epoch in range(num_epoch):
    epoch_start_time = time.time()
    # 返回当前时间的时间戳（1970纪元后经过的浮点秒数）
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train()  # 确保 model 是在 train model (开启 Dropout 等...)
    for i, data in enumerate(train_loader):
        # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
        # 一般用在 for 循环当中。
        optimizer.zero_grad()  # 用 optimizer 将 model 参数的 gradient 归零
        train_pred = model(data[0].to(device))  # 调用 model 的 forward 函數
        batch_loss = loss(train_pred, data[1].to(device))  # 计算 loss
        batch_loss.backward()  # 利用 back propagation 算出每个参数的 gradient
        optimizer.step()  # 以 optimizer 用 gradient 更新参数值

        train_acc += np.sum(
            np.argmax(train_pred.cpu().data.numpy(), axis=1) ==
        # argmax: 取出train_pred中元素最大值所对应的索引（下标）
        # 把train_pred放在CPU上，并从tensor转化为numpy形式
        # sum()输入参数带有axis时，将按照指定axis（轴）进行对应求和
            data[1].numpy())
        train_loss += batch_loss.item()

    model.eval()
    with torch.no_grad():
    # 数据不需要计算梯度，也不会进行反向传播
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].to(device))
            batch_loss = loss(val_pred, data[1].to(device))

            val_acc += np.sum(
                np.argmax(val_pred.cpu().data.numpy(), axis=1) ==
                data[1].numpy())
            val_loss += batch_loss.item()

    # 计算均值
    train_acc /= train_set.__len__()
    train_loss /= train_set.__len__()
    val_acc /= val_set.__len__()
    val_loss /= val_set.__len__()

    # 将结果 print 出来
    print(
        '[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f'
        % (epoch + 1, num_epoch, time.time() - epoch_start_time, train_acc,
           train_loss, val_acc, val_loss))

    # 记录最好的结果 并保存模型
    if val_acc > val_acc_best:
        val_acc_best = val_acc
        torch.save(model.state_dict(), 'model_best.pth.tar')
        print('Save model')

print('Best accuracy on validation set: %3.6f' % val_acc_best)
