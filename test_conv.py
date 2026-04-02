import torch
from Models.conv import Conv1d

conv = Conv1d(10, 10, 1)
print(conv.weight.mean().item(), conv.weight.std().item())
