import torch
import torch.nn as nn

# dropout 随机丢弃同时保持其他元素期望不变
dropout_layer = nn.Dropout(p=0.5)  # 丢弃概率为0.5

t1 = torch.Tensor([1, 2, 3])  # shape=(3,)
t2 = dropout_layer(t1)  # shape=(3,)
# 这里dropout丢弃为了保持期望不变，将其他部分变为原来的两倍
print(t2)  # output: tensor([2., 0., 6.])  # 第二个元素被丢弃了

dropout_layer2 = nn.Dropout(p=0.2)
t2 = dropout_layer2(t1)
print(t2)

# linear 线性变换 y=Wx+b， input->output
layer = nn.Linear(in_features=3, out_features=5, bias=True)
t1 = torch.Tensor([1, 2, 3])  # shape=(3,)

t2 = torch.Tensor([[1, 2, 3]])  # shape=(1,3)
# 这里应用的w和b是随机的，真是训练里会在optimizer上更新
output2 = layer(t2)  # shape=(1,5)
print(output2)

# view 改变张量形状，不改变数据
t = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])  # shape=(2,6)
t_view1 = t.view(3, 4)
print(t_view1)  # shape=(3,4)
t_view2 = t.view(4, 3)
print(t_view2)

# transpose 交换张量的维度
t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])  # shape=(2,3)
t1 = t1.transpose(0, 1)  # shape=(3,2)
print(t1)


# triu 上三角矩阵 default diagonal=0
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # shape=(3,3)
print(torch.triu(x))  # 上三角矩阵，主对角线及以上
print(torch.triu(x, diagonal=1))  # 上三角矩阵，主对角线以上


# reshape 改变张量形状，不改变数据
x = torch.arange(1, 7)  # tensor([1, 2, 3, 4, 5, 6])
y = torch.reshape(x, (2, 3))  # tensor([[1, 2, 3],[4, 5, 6]])
print(y)  # tensor([[1, 2, 3],[4, 5, 6]])
# 使用-1 自动推断
z = torch.reshape(x, (3, -1))  # tensor([[1, 2],[3, 4],[5, 6]])
print(z)  # tensor([[1, 2],[3, 4],[5, 6]])
