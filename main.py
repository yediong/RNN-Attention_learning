import torch.nn as nn
import numpy as np
from struct import unpack
import gzip
import torch.optim as optim
import torch
import matplotlib.pyplot as plt

epochs = 20
batch_size = 200
learning_rate = 0.01

def index_calculation(TP, FP, FN, TN):
    """
                    true
                阳性      阴性
    predict 阳性 TP       FP
            阴性 FN       TN
    """
    Sum = TP + FP + FN + TN
    Accuracy = (TP + TN) / Sum
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = 2 * (Precision * Recall) / (Precision + Recall)

    print(f'Accuracy={Accuracy} \nPrecision={Precision} \nRecall={Recall} \nF1={F1}')


class Fully_Connected(nn.Module):
    def __init__(self, input, hidden1, hidden2, output):
        super(Fully_Connected, self).__init__()
        self.fc1 = nn.Linear(input, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def __read_image(path):
    with gzip.open(path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        img = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)
    return img


def __read_label(path):
    with gzip.open(path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        lab = np.frombuffer(f.read(), dtype=np.uint8)
        # print(lab[1])
    return lab

def __normalize_image(image):
    img = image.astype(np.float32) / 255.0
    return img

def load_mnist(x_train_path, y_train_path, x_test_path, y_test_path, normalize=True, one_hot=True):
    image = {
        'train': __read_image(x_train_path),
        'test': __read_image(x_test_path)
    }

    label = {
        'train': __read_label(y_train_path),
        'test': __read_label(y_test_path)
    }

    if normalize:
        for key in ('train', 'test'):
            image[key] = __normalize_image(image[key])

    return (image['train'], label['train']), (image['test'], label['test'])


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # 读取数据集
    x_train_path = './Mnist/train-images-idx3-ubyte.gz'
    y_train_path = './Mnist/train-labels-idx1-ubyte.gz'
    x_test_path = './Mnist/t10k-images-idx3-ubyte.gz'
    y_test_path = './Mnist/t10k-labels-idx1-ubyte.gz'
    (x_train, y_train), (x_test, y_test) = load_mnist(x_train_path, y_train_path, x_test_path, y_test_path)

    # 指标计算
    # index_calculation(80, 20, 40, 60)

    # 全连接网络
    net = Fully_Connected(28 * 28, 128, 256, 10)

    # 用随机梯度将所有参数和反向传播器的梯度缓冲区归零
    net.zero_grad()

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 定义优化器
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)

    # 转换数据类型为Tensor
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    losses = []

    for epoch in range(epochs):
        running_loss = 0.0
        for i in range(0, len(x_train), batch_size):
            # 获取当前batch的数据和标签
            inputs = x_train_tensor[i:i + batch_size]
            labels = y_train_tensor[i:i + batch_size]

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = net(inputs)

            # 计算损失
            loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()

            # 参数更新
            optimizer.step()

            # 打印统计信息
            running_loss += loss.item()
            # if i % 1000 == 999:  # 每1000个mini-batches打印一次
            #     print('[%d, %5d] loss: %.3f' %
            #           (epoch + 1, i + 1, running_loss / 1000))
            #     running_loss = 0.0

        epoch_loss = running_loss / (len(x_train) / batch_size)
        losses.append(epoch_loss)

        # 打印每个epoch的损失值
        print('Epoch [%d/%d], Loss: %.4f' % (epoch + 1, epochs, epoch_loss))

    print('Finished Training')

    # 绘制损失曲线
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.show()

    net.eval()
    correct = 0
    total = 0

    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    with torch.no_grad():   #关闭梯度计算
        for i in range(0, len(x_test), batch_size):
            inputs = x_test_tensor[i:i + batch_size]
            labels = y_test_tensor[i:i + batch_size]

            # 获取模型预测结果
            outputs = net(inputs)

            # 计算预测正确的样本数量
            max , predicted = torch.max(outputs, 1)  #概率最高即置信度最高的类别即为预测。predicted为对应索引，故
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 打印测试集准确率
    print('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))
#    x_train=x_train.view(-1,784)
#    output = net(x_train)
