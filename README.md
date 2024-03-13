# *learning notes* by 叶庭宏
## *Day1* （2024.3.12）


今天的进度为完成了第一题的主要部分，还差代入指标计算函数和探索参数对模型表现的影响。空余时间为第5、6节课，11、12节课和半夜。以下是对学习过程的回顾和总结。

---
### 1. 评测指标
上周模式识别刚讲了二分类情况下的评测指标，复习了一下。计算评测指标代码没啥东西，套公式就完事了。
主要在于理解每个指标的意义：
- Accurancy：整体预测正确情况占比。
- Precision：不要求整体准确率高，只要positive类中精度高。“冤假错案成本高，漏网之鱼成本低”。
- Recall：不管是不是真的positive，都要抓出来。宁可错杀一千，不可放过一个。
- F1：综合考虑Accurancy和Recall，达到两者的最大平衡。

---
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

---
### 3. 分类任务
先在网上学习了MNIST数据集读取的方法，最终提取出测试集和训练集的img和label共四个数组。调用全连接网络输入各级神经元个数（28 * 28, 128, 256, 10）并存入模型。分批训练，步骤如下：**前向传播 -> 计算loss值 -> 反向传播 -> 用优化器更新参数**，以此循环若干epoch，并将每次迭代的loss值可视化。

后用测试集计算准确率Accurancy。选择outputs中的最大值（概率最高即置信度最高，视为预测结果），第二个参数predicted为最大值对应索引，即label。最终计算预测正确的样本数，除以测试集总样本数即为Accurancy。今日测试结果如下：

<img src=loss_old.png width=500> <img src=result.png width=400>

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

___

<img src=Training_Accuracy.png width=500> <img src=Loss.png width=500>


___

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

    <img src=epoch5_acc.png width=400> <img src=epoch5_loss.png width=400> 

   - epoch=200时

    <img src=epoch200_acc.png width=400> <img src=epoch200_loss.png width=400> 

