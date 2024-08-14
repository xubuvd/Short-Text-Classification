""" Created by minhnq """
import torch
a = torch.Tensor([1,2,3])
values, indices = torch.max(a, 0)
print(values)
print(indices)