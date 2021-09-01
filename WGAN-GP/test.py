# -*- coding: utf-8 -*-
# @Time    : 2021/8/24 20:47
# @Author  : Wu Weiran
# @File    : test.py
import torch
from torch import autograd

x = torch.rand(3, 4)
x.requires_grad_()
print(x)

y = x[:, 0] + x[:, 1]
print(y)
grad = autograd.grad(
    outputs=y,
    inputs=x,
    grad_outputs=torch.ones_like(y)
)[0]
print(grad)
