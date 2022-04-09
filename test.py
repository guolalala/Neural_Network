'''
author: Bodan Chen
Date: 2022-02-16 16:30:29
LastEditors: Bodan Chen
LastEditTime: 2022-02-18 01:59:53
Email: 18377475@buaa.edu.cn
'''
#Code: https://github.com/MorvanZhou/PyTorch-Tutorial
import torch
import numpy as np


# details about math operation in torch can be found in: http://pytorch.org/docs/torch.html#math-operations
np_data=np.arange(6)
torch_data = torch.from_numpy(np_data)
tensor2array=torch_data.numpy()
print(
    '\nnum array:',np_data,
    '\ntorch tensor:',torch_data,
    '\ntensor to array:',tensor2array,
)

# np.abs(data)
# torch.abs(data)