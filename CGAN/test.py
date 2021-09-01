# -*- coding: utf-8 -*-
# @Time    : 2021/8/25 20:29
# @Author  : Wu Weiran
# @File    : test.py
import torch
import numpy as np

# x = torch.randn(2, 3)
#
# y = torch.cat((x, x, x), 0)
#
# z = torch.cat((x, x, x), 1)
#
# t = torch.cat((x, x, x), -1)  # dim = -1表示负索引，即倒数第一个dim
#
# print(x)
# print(y)
# print(z)
# print(t)

# a = np.random.randint(0, 10, 64)
# b = torch.randint(10, (64,))
# print(a)
# print(b)


# label = torch.ones((4, 1), dtype=torch.int8)*1
# label2 = torch.ones((4, 1), dtype=torch.int8)*2
# a = torch.cat((label, label2), dim=0)
# print(a)