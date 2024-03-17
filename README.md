# *learning notes* by 叶庭宏
## *Day1* （2024.3.12）


今天的进度为完成了第一题的主要部分，还差代入指标计算函数和探索参数对模型表现的影响。空余时间为第5、6节课，11、12节课和半夜。以下是对学习过程的回顾和总结。

### 1. 评测指标
上周模式识别刚讲了二分类情况下的评测指标，复习了一下。计算评测指标代码没啥东西，套公式就完事了。
主要在于理解每个指标的意义：
- Accurancy：整体预测正确情况占比。
- Precision：不要求整体准确率高，只要positive类中精度高。“冤假错案成本高，漏网之鱼成本低”。
- Recall：不管是不是真的positive，都要抓出来。宁可错杀一千，不可放过一个。
- F1：综合考虑Accurancy和Recall，达到两者的最大平衡。

### 2. 全连接层
网上找资料了解了一下用nn.Linear构建全连接层和前向传播的原理和方法。
- 全连接层：
```
class Fully_Connected(nn.Module):
    def __init__(self, input, hidden1, hidden2, output):
        super(Fully_Connected, self).__init__()
        self.fc1 = nn.Linear(input, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output)
```
传入参数为输入层、中间各层、输出层的神经元个数。main中输出为10是因为label共有10种。

- 前向传播函数
```
def forward(self, x):
    x = self.fc1(x)
    x = self.fc2(x)
    x = self.fc3(x)
    return x
```

### 3. 分类任务
先在网上学习了MNIST数据集读取的方法，最终提取出测试集和训练集的img和label共四个数组。调用全连接网络输入各级神经元个数（28 * 28, 128, 256, 10）并存入模型。分批训练，步骤如下：**前向传播 -> 计算loss值 -> 反向传播 -> 用优化器更新参数**，以此循环若干epoch，并将每次迭代的loss值可视化。

后用测试集计算准确率Accurancy。选择outputs中的最大值（概率最高即置信度最高，视为预测结果），第二个参数predicted为最大值对应索引，即label。最终计算预测正确的样本数，除以测试集总样本数即为Accurancy。今日测试结果如下：

<img src=picture/loss_old.png width=500> <img src=picture/result.png width=400>

___
## *Day2* (2024.3.13)
今天正好生日，做的不多，第一题基本完成了。
### 1. 对第一题的代码进行了如下修改：
- 修改了评价指标函数，使其适用于多分类问题。最终打印各参数的均值。
- 将测试过程并入epoch循环，便于计算Accuracy值及其变化曲线。
- 将train和test的loss变化曲线放在一个图中，结果如下。

评价指标：
> Accuracy=0.875 
>
> Precision=0.8715983553483554 
>
> Recall=0.8616165413533835 
>
> F1=0.8623119627497193

四个指标值都还行。


<img src=picture/Training_Accuracy.png width=500> <img src=picture/Loss.png width=500>




### 2. 探索参数对模型表现的影响

- 各层神经元个数
保持*epochs=20, batch_size=200, learning_rate=0.01*
一开始尝几十几十地变，感觉没有太明显的区别，干脆尝试极端情况。
    - (1,256)：第一层神经元个数极小，此时Loss值为1.6，评价指标均未超过0.5。
    - (1,2048)：加大两层的差，结果更为答辩
    - (4,256)：稍微增大第一层，效果显著，正常多了
    - (4,2048)：继续加大第二层，结果变化不大，但是epoch=1时loss值低了很多
    - (256,1)：和(1,256)区别不大

*conclusion:*

    1. 神经元个数越多，训练越慢
    2. 增大第二层的神经元个数可以显著降低loss初值（效果似乎强于增大第一层）
    3. 层数过小时结果很差，表现为评估指标值都很低，但稍微增大（>=4）就基本正常了

- 神经元层数

| 层数  |  结果 |
| ------- | --------- |
| 1层（128）| Accuracy=0.85，loss=0.32 |
| 2层（128，256）| Accuracy=0.87，loss=0.30 |
| 3层（128，256，256）| Accuracy=0.87，loss=0.29 |
| 4层（128，256，256，1024）| Accuracy=0.87，loss=0.28 |

感觉层数增大对结果没有太大影响，但总体有微小的优化。

- 训练步数
   epochs过大，曲线趋于平缓，后期准确率提升缓慢，时间成本增加。epochs过小，迭代次数不够，可能导致loss值偏大，准确率低。折中即可。
   
   - epoch=5时

    <img src=picture/epoch5_acc.png width=400> <img src=picture/epoch5_loss.png width=400> 

   - epoch=200时

    <img src=picture/epoch200_acc.png width=400> <img src=picture/epoch200_loss.png width=400> 

---
## *Day3* (2024.3.14)
下午了解了RNN原理并尝试实现，维度匹配耗费了一点时间。晚上尝试提升训练速度（最难绷的一集）。

### 1. 对原理的理解
将一个输入样本按照n个时间步，分割成n次循环，每一次的隐藏层由输入和上一个时间步所得的隐藏层共同计算得到，输出同理。

### 2. 实现
和第一道的主要区别在于网络内部的结构差异，并且多了一个分割input和拼接的步骤。外部的训练测试和数据读取函数都没有太大区别。

一开始想尝试看完原理直接自己写net，有助于深刻理解rnn的结构。于是就有了下面这一坨forward：
```
    def forward(self, inputs_batch, hidden_num):
        # 存放一个batch的所有predicted
        outputs_batch = []
        # inputs_batch为每批的输入，shape为(batch_size, 28*28)，此处拆成64个一维向量
        for inputs in inputs_batch:
            # 存放单个样本的结果
            outputs = []
            hidden = torch.zeros(hidden_num)
            # hidden = hidden.to(inputs.device)

            # 为了满足rnn条件，把一个inputs分割成28个输入
            split_inputs = [inputs[i * 28:(i + 1) * 28] for i in range(28)]

            for input in split_inputs:
                combined = torch.cat((input, hidden), dim=0)
                hidden = self.relu(self.hidden_layer(combined))
                output = self.output_layer(combined)
                outputs.append(output)
            # 单个样本结果取均值，即predicted
            output_1d = torch.mean(torch.stack(outputs), dim=0)
            # 将单样本结果加入
            outputs_batch.append(output_1d)

        # 将多个向量合并成一个张量
        return torch.stack(outputs_batch)
```
我的思路还是太c语言了，先套了第一层循环用来抽取单个样本，再套了第二层循环用来给拆分后的样本拼接hidden层，最后把结果缝合成一个（batch_size,1）的输出。

此外，维度的问题调试了一段时间，最后捋清楚了。

耗时最多的是训练速度慢的问题。刚能运行的时候一个epoch要跑差不多十分钟，这样就很难试出合适的训练参数。于是我进行了以下的尝试：
- 换优化器：把SGD换成了Adam，毫无卵用
- 换激活层：先加了tanh，后改成relu，无效果
- 调整梯度清零的位置：意义不大
- cpu换gpu：想用gpu跑，但是当前虚拟环境里的pytorch是cpu版本，只能删了重下。装好后测试显示gpu没开，检查半天发现是cuda版本和pytorch不匹配。最后终于能用gpu了，速度还是没什么区别。
- 数据读取的问题：没用pytorch内置的dataloader，可能对速度有一定影响。但是改过来会对后面的代码造成较大的改动，且第一题用同样的数据读取方法没出问题。
- 提升硬件性能：回去以后插上电开野兽模式跑了一晚上，醒过来看发现是一坨：

<img src=picture/acc_bad.png width=400> <img src=picture/loss_bad.png width=400>

---
## *Day4* (2024.3.15)
先处理rnn训练慢的问题。

### 1. 优化rnn结构
最终确定问题出在网络结构上。于是我去网上找了一些rnn前向传播函数的示例学习了一下，发现问题在于之前的forward里循环层数太多（c课设后遗症），而我对python语法还不够熟，没想到可以直接对高维数组操作。于是修改代码如下：

```
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
```

思路是把输入的二维数组重塑成(batch_size,28,28)的三维数组，以第二维（也就是按时间步）为基准循环，相当于从batch_size个二维数组中，抽出每个数组的第i行（共28行），每个后面拼接一个hidden，以此循环。这样就减少了一层将batch拆成单个样本的循环。

### 2. 完成第二题
速度上来以后很快调好了参数，训练结果如下：
>Accuracy=0.915 
>
>Precision=0.9030176767676767 
>
>Recall=0.9013231608432847 
>
>F1=0.9002948928934773

可视化：

<img src=picture/acc_good2.png width=400> <img src=picture/loss_good2.png width=400>

---
## *Day5* (2024.3.16)
### 1. 学习self attention和multi-head attention机制
看了算法题pdf里推荐的两个视频，我的理解如下：

attention，以nlp为例，就相当于在一句话中一个单词和其他单词的关联程度，通过关联程度的计算，能够判断这个单词对应的value（即含义）。

从最基础的self attention出发：输入信息分别乘以不同矩阵转化成Q，K，V三个矩阵，再用Q和K进行矩阵乘法获得一句话里多个单词之间的关系矩阵（即单词之间相对的注意力权重），最后将权重矩阵经过softmax处理后乘以V，就可以得到最终输出。

整理笔记：

<img src=picture/attention原理笔记.jpg width=750>

- **Multi-Head**：原始的Q,K,V进行影分身，得到相同的n组Q,K,V，再各自算出对应的权重矩阵，相当于在中间增加了一层维度。

- **对于Q,K,V参数的理解**：
  - Q即query，理解为输入的查询请求，希望得到对应的value。
  - K即key，类似于python里的键值对，映射到value，通过与Q比照（距离的远近），算出Q对应的value（感觉K有点类似于参考样本？）
  - V即value，目标值。
  
- **碰到的难点**：感觉主要是矩阵维数的匹配有点绕，要理解每个维度代表的意义和乘法的作用。


### 2. 实现MHA
核心代码：

```
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
```

权重矩阵：

<img src=picture/MHA.png width=750>
