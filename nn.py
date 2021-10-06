import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    # 必须继承自nn.moddule类，是nn最基本的类，可以是一个tensor或者是一个tensor的集合
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, (5, 5))  # 输入通道数，输出通道数
        self.conv2 = nn.Conv2d(6, 16, (5, 5))
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


# load data
transform = torchvision.transforms.ToTensor()  # 定义数据预处理方式：转换 PIL.Image 成 torch.FloatTensor

train_data = torchvision.datasets.MNIST(root="D:\\dataset\\minist\\train",  # 数据目录，这里目录结构要注意。
                                        train=True,  # 是否为训练集
                                        transform=transform,  # 加载数据预处理
                                        download=False)  # 是否下载
test_data = torchvision.datasets.MNIST(root="D:\\dataset\\minist\\train",
                                       train=False,
                                       transform=transform,
                                       download=False)

# 数据加载器:组合数据集和采样器
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=128, shuffle=False)



if __name__ == '__main__':
    net = LeNet()
    print('net:{}'.format(net))
    # define loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 若检测到GPU环境则使用GPU，否则使用CPU
    net = LeNet().to(device)  # 实例化网络，有GPU则将网络放入GPU加速
    loss_fuc = nn.CrossEntropyLoss()  # 多分类问题，选择交叉熵损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # 选择SGD，学习率取0.001
    # 开始训练
    EPOCH = 8  # 迭代次数
    for epoch in range(EPOCH):
        sum_loss = 0
        # 数据读取
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # 有GPU则将数据置入GPU加速

            # 梯度清零
            optimizer.zero_grad()

            # 传递损失 + 更新参数
            output = net(inputs)
            loss = loss_fuc(output, labels)
            loss.backward()
            optimizer.step()

            # 每训练100个batch打印一次平均loss
            sum_loss += loss.item()
            if i % 100 == 99:
                print('[Epoch:%d, batch:%d] train loss: %.03f' % (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0

        correct = 0
        total = 0

        for data in test_loader:
            test_inputs, labels = data
            test_inputs, labels = test_inputs.to(device), labels.to(device)
            outputs_test = net(test_inputs)
            _, predicted = torch.max(outputs_test.data, 1)  # 输出得分最高的类
            total += labels.size(0)  # 统计50个batch 图片的总个数
            correct += (predicted == labels).sum()  # 统计50个batch 正确分类的个数

        print('第{}个epoch的识别准确率为：{}%'.format(epoch + 1, 100 * correct.item() / total))

    # 模型保存
    torch.save(net.state_dict(), 'D:\\gitcode\\AE\\model\\ckpt.mdl')

    # 模型加载
    # net.load_state_dict(torch.load('ckpt.mdl'))

