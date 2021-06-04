import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10(root='./data', 	  # 数据集存放目录
										 train=True,		  # 表示是数据集中的训练集
                                         download=True,  	  # 第一次运行时为True，下载数据集，下载完成后改为False
                                         transform=transform) # 预处理过程
# 加载训练集，实际过程需要分批次（batch）训练
train_loader = torch.utils.data.DataLoader(train_set, 	   # 导入的训练集
										   batch_size=50,  # 每批训练的样本数
                                           shuffle=False,  # 是否打乱训练集
                                           num_workers=0)  # 使用线程数，在windows下设置为0
# 导入10000张测试图片
test_set = torchvision.datasets.CIFAR10(root='./data',
										train=False,	   # 表示是数据集中的测试集
                                        download=False,
                                        transform=transform)
# 加载测试集
test_loader = torch.utils.data.DataLoader(test_set,
										  batch_size=10000, # 每批用于验证的样本数
										  shuffle=False,
                                          num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 获取测试集中的图像和标签，用于accuracy计算
test_data_iter = iter(test_loader)
test_image, test_label = test_data_iter.next()
