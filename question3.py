import torch.nn as nn
import torch
import math
import matplotlib.pyplot as plt


def visualize(attention_weights, name):
    num_head, seq_len, _ = attention_weights.shape
    plt.figure(figsize=(10, 6 * num_head))
    for i in range(num_head):
        plt.subplot(num_head, 1, i + 1)
        plt.imshow(attention_weights[i], cmap='hot', interpolation='nearest')
        plt.xlabel('Source')
        plt.ylabel('Target')

        if name == 1:
            plt.title(f'MHA Attention Weights (Head {i + 1})')
        elif name == 2:
            plt.title(f'MQA Attention Weights (Head {i + 1})')
        else:
            plt.title(f'GQA Attention Weights (Head {i + 1})')

        plt.colorbar()
    plt.tight_layout()
    plt.show()


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


class MultiQueryAttention(nn.Module):
    def __init__(self, input_dim, dim_k, dim_v, num_head):
        super(MultiQueryAttention, self).__init__()
        assert dim_k % num_head == 0
        assert dim_v % num_head == 0
        self.dim_k = dim_k
        self.dim_v = dim_v
        # 创建多个q，引入不同的查询
        self.q = nn.ModuleList([nn.Linear(input_dim, dim_k) for _ in range(num_head)])
        self.k = nn.Linear(input_dim, dim_k)
        self.v = nn.Linear(input_dim, dim_v)
        self.num_head = num_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # 把q堆叠，多出一个维度1
        Q = torch.stack([q(x) for q in self.q], dim=1)
        # 在1的维度上扩展一个维度num_head
        K = self.k(x).unsqueeze(1).expand(-1, self.num_head, -1, -1)
        V = self.v(x).unsqueeze(1).expand(-1, self.num_head, -1, -1)

        attention = self.softmax(torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(self.dim_k)).transpose(-2, -1)
        output = torch.matmul(attention, V).reshape(-1, x.shape[1], x.shape[2])
        return attention, output


class GroupQueryAttention(nn.Module):
    def __init__(self, input_dim, dim_k, dim_v, num_group, num_head):
        super(GroupQueryAttention, self).__init__()
        assert dim_k % num_group == 0
        assert dim_v % num_group == 0
        assert dim_k % num_head == 0
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_group = num_group
        self.num_head = num_head

        # 创建多个组，每个组包含多个头
        self.groups = nn.ModuleList([
            MultiHeadAttention(input_dim, dim_k // num_head, dim_v // num_head, num_head)
            for _ in range(num_group)
        ])

    def forward(self, x):
        attention_outputs = []
        group_outputs = []
        for group in self.groups:
            group_attention, group_output = group(x)
            attention_outputs.append(group_attention.unsqueeze(1))
            group_outputs.append(group_output.unsqueeze(1))

        # 不同group取均值汇总
        attention = torch.mean(torch.cat(attention_outputs, dim=1), dim=1)
        output = torch.mean(torch.cat(group_outputs, dim=1), dim=1)
        return attention, output


if __name__ == "__main__":
    batch_size = 1
    input_dim = 4
    seq_len = 6
    input = torch.randn(batch_size, seq_len, input_dim)

    # 定义注意力机制参数
    dim_k = 8
    dim_v = 8
    nums_head = 2
    num_group = 2

    # 创建多头注意力机制模型
    mha = MultiHeadAttention(input_dim, dim_k, dim_v, nums_head)
    mqa = MultiQueryAttention(input_dim, dim_k, dim_v, nums_head)
    gqa = GroupQueryAttention(input_dim, dim_k, dim_v, num_group, nums_head)

    # 使用模型计算注意力权重
    attention_mha, attention_output_mha = mha(input)
    attention_weights_mha = attention_mha.squeeze(0).cpu().detach().numpy()

    attention_mqa, attention_output_mqa = mqa(input)
    attention_weights_mqa = attention_mqa.squeeze(0).cpu().detach().numpy()

    attention_gqa, attention_output_gqa = gqa(input)
    attention_weights_gqa = attention_gqa.squeeze(0).cpu().detach().numpy()

    # 可视化注意力权重
    visualize(attention_weights_mha, 1)
    visualize(attention_weights_mqa, 2)
    visualize(attention_weights_gqa, 3)
