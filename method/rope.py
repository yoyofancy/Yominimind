import torch

x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
y = torch.tensor([10, 20, 30, 40, 50])

condition = x > 3

result = torch.where(condition, x, y)

print(result)


# arange --- 生成等差数列

t = torch.arange(0, 10, 2)
print(t)
t = torch.arange(5, 0, -1)
print(t)


# outer product
v1 = torch.tensor([1, 2, 3])
v2 = torch.tensor([4, 5, 6])
result = torch.outer(v1, v2)
print(result)


# cat --- 拼接
t1 = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
t2 = torch.tensor([[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]])
print(t1.shape)
result = torch.cat((t1, t2), dim=0)
print(result)

result = torch.cat((t1, t2), dim=1)
print(result)

result = torch.cat((t1, t2), dim=-1)
print(result)


# unsqueeze --- 在指定位置插入维度为1的维度
t1 = torch.tensor([1, 2, 3, 4])
t2 = t1.unsqueeze(0)
print(t1.shape)
print(t2.shape)
print(t2)
