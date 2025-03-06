import torch.nn as nn
import numpy as np
from struct import unpack
import gzip
import torch.optim as optim
import torch
import matplotlib.pyplot as plt

epochs = 100
batch_size = 200
learning_rate = 0.001
net_size = 128


# 沿用第一题的指标计算
def index_calculation(matrix_np, dim):
    matrix = torch.from_numpy(matrix_np)
    diagonal = torch.diag(matrix)
    diag_sum = torch.sum(diagonal)
    total_sum = torch.sum(matrix)

    # 准确率
    Accuracy = diag_sum / total_sum

    # 每类的precision,recall和F1 score
    Precision = diagonal / matrix.sum(1)
    Recall = diagonal / matrix.sum(0)
    F1 = 2 * (Precision * Recall) / (Precision + Recall)

    # 平均值
    Precision_mean = Precision.mean()
    Recall_mean = Recall.mean()
    F1_mean = F1.mean()

    print(f'Accuracy={Accuracy} \nPrecision={Precision_mean} \nRecall={Recall_mean} \nF1={F1_mean}')


class rnn(nn.Module):
    def __init__(self, input_num, hidden_num, output_num):
        super(rnn, self).__init__()
        self.hidden_layer = nn.Linear(input_num + hidden_num, hidden_num)
        self.output_layer = nn.Linear(input_num + hidden_num, output_num)
        self.relu = nn.ReLU()

    def forward(self, inputs_batch, hidden_num):
        # 把二维的inputs(batch_size,784)重塑成(batch_size,28,28)的三维数组
        inputs = inputs_batch.squeeze().view(-1, 28, 28)

        # 在每个epoch中重置一个新的隐藏层(二维)
        hidden = torch.zeros(batch_size, hidden_num)

        # gpu
        hidden = hidden.to(inputs.device)

        # 以第二维（也就是按时间步）循环，相当于从batch_size个二维数组中，抽出每个数组的第i行（共28行），每个后面拼接一个hidden
        for i in range(inputs.shape[1]):
            input = inputs[:, i, :]
            combined = torch.cat((input, hidden), dim=1)  # 横着拼
            hidden = self.relu(self.hidden_layer(combined))  # 一开始用的是tanh，但网上说relu更好
            outputs = self.output_layer(combined)

        return outputs

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


if __name__ == '__main__':
    # 读取数据集
    x_train_path = './fashion-mnist/train-images-idx3-ubyte (1).gz'
    y_train_path = './fashion-mnist/train-labels-idx1-ubyte (1).gz'
    x_test_path = './fashion-mnist/t10k-images-idx3-ubyte (1).gz'
    y_test_path = './fashion-mnist/t10k-labels-idx1-ubyte (1).gz'
    (x_train, y_train), (x_test, y_test) = load_mnist(x_train_path, y_train_path, x_test_path, y_test_path)

    net = rnn(28, net_size, 10)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 定义优化器
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)  # Adam梯度下降
    # optimizer = optim.SGD(net.parameters(), lr=learning_rate)

    # 转换数据类型为Tensor
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_losses = []
    test_losses = []
    accuracies = []

    # gpu
    if torch.cuda.is_available():
        net = net.cuda()
        criterion = criterion.cuda()
        x_train_tensor = x_train_tensor.cuda()
        y_train_tensor = y_train_tensor.cuda()
        x_test_tensor = x_test_tensor.cuda()
        y_test_tensor = y_test_tensor.cuda()
    else:
        print('gpu not working')

    for epoch in range(epochs):
        # 初始化
        running_loss = 0.0
        correct = 0
        total = 0

        for i in range(0, len(x_train), batch_size):
            # 获取当前batch的数据和标签
            inputs = x_train_tensor[i:i + batch_size]
            labels = y_train_tensor[i:i + batch_size]

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = net(inputs, net_size)

            # 计算损失
            loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()

            # 参数更新
            optimizer.step()

            # 打印统计信息
            running_loss += loss.item()

            # 计算训练中的准确率
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # print('true = %.4f, i = %d' % (correct / total, i))

        epoch_loss = running_loss / (len(x_train) / batch_size)
        train_losses.append(epoch_loss)

        accuracy = 100 * correct / total
        accuracies.append(accuracy)

        # 打印每个epoch的损失值
        print('Epoch [%d/%d], Loss: %.4f' % (epoch + 1, epochs, epoch_loss))

        # 测试集，启动
        net.eval()

        with torch.no_grad():  # 关闭梯度计算
            running_test_loss = 0.0
            for i in range(0, len(x_test), batch_size):
                inputs = x_test_tensor[i:i + batch_size]
                labels = y_test_tensor[i:i + batch_size]

                # 获取模型预测结果
                outputs = net(inputs, net_size)
                # 计算loss
                loss = criterion(outputs, labels)
                running_test_loss += loss.item()

            epoch_test_loss = running_test_loss / (len(x_test) / batch_size)
            test_losses.append(epoch_test_loss)

    print('Finished Training')

    # 参数初始化
    dim = len(torch.unique(y_test_tensor))  # 读取维数
    correct = 0
    total = 0
    class_correct = [0] * dim
    class_total = [0] * dim
    matrix = np.zeros([dim, dim])

    # 计算最后一次预测正确的样本数量
    _, predicted = torch.max(outputs, 1)  # 概率最高即置信度最高的类别即为预测。predicted为对应索引，故

    # 数据统计，用于计算评价指标
    for label, pred in zip(labels, predicted):
        class_correct[label] += int(label == pred)
        class_total[label] += 1
        matrix[pred][label] += 1

    # 打印测试集评价指标
    index_calculation(matrix, dim)

    # 绘制训练准确率曲线
    plt.plot(accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Curve')
    plt.show()

    # 绘制损失曲线
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.show()
