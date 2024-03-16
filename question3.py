import torch.nn as nn
import numpy as np
import torch
import math
import matplotlib.pyplot as plt


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, dim_k, dim_v, num_head):
        super(MultiHeadAttention, self).__init__()
        # 确保k和v的维度能整除头数
        assert dim_k % num_head == 0
        assert dim_v % num_head == 0
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.q = nn.Linear(input_dim, dim_k)
        self.k = nn.Linear(input_dim, dim_k)
        self.v = nn.Linear(input_dim, dim_v)
        self.num_head = num_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # (batch_size, seq_len（一句话有几个单词）, 头数, 维度（身高、体重）)，再把seq_len和dim放最后
        Q = self.q(x).view(-1, x.shape[1], self.num_head, self.dim_k // self.num_head).permute(0, 2, 1, 3)
        K = self.k(x).view(-1, x.shape[1], self.num_head, self.dim_k // self.num_head).permute(0, 2, 1, 3)
        V = self.v(x).view(-1, x.shape[1], self.num_head, self.dim_v // self.num_head).permute(0, 2, 1, 3)

        attention = self.softmax(torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(self.dim_k)).transpose(-2, -1)
        output = torch.matmul(attention, V).reshape(-1, x.shape[1], x.shape[2])
        return attention, output

    def visualize(self, attention_weights):
        num_head, seq_len, _ = attention_weights.shape
        plt.figure(figsize=(10, 6 * num_head))
        for i in range(num_head):
            plt.subplot(num_head, 1, i + 1)
            plt.imshow(attention_weights[i], cmap='hot', interpolation='nearest')
            plt.xlabel('Source')
            plt.ylabel('Target')
            plt.title(f'Attention Weights (Head {i + 1})')
            plt.colorbar()
        plt.tight_layout()
        plt.show()


if __name__== "__main__":
    batch_size = 1
    input_dim = 4
    seq_len = 5
    input = torch.randn(batch_size,seq_len,input_dim)

    # 定义注意力机制参数
    dim_k = 8
    dim_v = 8
    nums_head = 2

    # 创建多头注意力机制模型
    mha = MultiHeadAttention(input_dim, dim_k, dim_v, nums_head)

    # 使用模型计算注意力权重
    attention,attention_output = mha(input)
    attention_weights = attention.squeeze(0).cpu().detach().numpy()

    # 可视化注意力权重
    mha.visualize(attention_weights)