import torch

a = torch.randn(5,5)
print(a)
b= torch.topk(a, k=2, dim=1).values
c= b.sum(dim=1)
#行和越大，越可能是类中心，进而计算一个较小的t值
t = 1-0.9 * (c-c.min())/(c.max()-c.min())
scores = a / t.view(-1,1)

a = torch.randn(5)
