import torch
import numpy as np

x = torch.tensor(2.0, requires_grad=True)
f = torch.exp(-x**2 -2*x - torch.sin(x))

f.backward()


print('function')
print(x.grad)


def manual(x):
    x1 = -x**2
    x2 = -2*x
    x3 = -np.sin(x)
    f = np.exp(x1+x2+x3)
    dfdx = f * (-2*x -2 - np.cos(x))
    return dfdx

print('manual')
print(manual(2))

# print(torch.cos(x), np.cos(2))