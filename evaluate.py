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


# 读取数据集文件;
# 文件名格式为label+id，即"1_2.jpg"表示label为1，id为2的jpg图片;
def readfile(path):
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file_name in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file_name))
        x[i, :, :] = cv2.resize(img, (128, 128))
        y[i] = int(file_name.split("_")[0])
    return x, y


# 设置随机数种子
setup_seed(1896)

# 分別将 training set、validation set 用 readfile 函数读进来;
workspace_dir = './food-3'
print("Reading data")
val_x, val_y = readfile(os.path.join(workspace_dir, "validation"))
print("Size of validation data = {}".format(len(val_x)))

# 验证集图片预处理
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])


# 构建验证集;
class ImgDataset(Dataset):
    def __init__(self, x, y, transform):
        self.x = x
        # label is required to be a LongTensor
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
val_set = ImgDataset(val_x, val_y, val_transform)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# 检测是否能够使用GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 搭建模型 model;
model = Classifier().to(device)

# 加载参数
model.load_state_dict(torch.load('model_best.pth.tar', map_location='cpu'))

val_acc = 0.0

model.eval()
with torch.no_grad():
    for i, data in enumerate(val_loader):
        val_pred = model(data[0].to(device))
        val_acc += np.sum(
            np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())

val_acc /= val_set.__len__()
print('Accuracy on validation set: %3.6f' % val_acc)


# 导出为 onnx 格式文件, 用于后续在华为云上进行推理
def convert():
    dummy_input = torch.randn(1, 3, 128, 128).to(device)
    torch.onnx.export(model,
                      dummy_input,
                      'model_best.onnx',
                      input_names=['input'],
                      output_names=['output'],
                      opset_version=11)