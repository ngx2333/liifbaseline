import torch


x=torch.rand(1,2,4)
x=x.repeat(3,1,1)
print(x.shape)