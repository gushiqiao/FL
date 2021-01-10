要求：
Python3
pytorch
Torchvision

数据：

手动下载训练和测试数据集，否则它们将自动从Torchvision数据集下载。

实验在Mnist和Cifar上进行。

方法：

运行baseline实验，请执行以下操作：

python src/federated_standard.py 

运行cmfl实验，请执行以下操作：

python src/federated_cmfl.py 

运行random mask实验，请执行以下操作：

python src/federated_mask.py 

运行sample实验，请执行以下操作：

python src/federated_sample.py 

运行ours实验，请执行以下操作：

python src/federated_chafen.py 

您可以更改其他参数的默认值以模拟不同的条件。请参阅选项部分。

选项：

解析为实验的各种参数的默认值在中给出options.py。

--dataset: 默认值：“ mnist”。选项：“ mnist”，“ cifar”

--model: 默认值：“ mlp”。选项：“ mlp”，“ cnn”

--gpu: 默认值：无（在CPU上运行）。也可以设置为特定的GPU ID。

--epochs: 训练轮数。

--lr: 学习率默认设置为0.01。

--verbose: 详细的日志输出。默认情况下处于激活状态，设置为0以禁用。

--seed: 随机种子。默认设置为1。

--iid: 在用户之间分发数据。默认设置为IID。对于非IID设置为0。

--num_users:用户数。默认值为100。

--frac: 用于联合更新的用户分数。默认值为0.1。

--local_ep:每个用户中的本地培训时期数。默认值是10。

--local_bs:每个用户中本地更新的批处理大小。默认值是10。

--unequal: 在非iid设置中使用。选择将数据平均或不平均地分配给用户。默认设置为0，表示均等分割。设置为1表示不相等的拆分。
