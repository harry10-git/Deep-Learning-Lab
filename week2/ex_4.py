import torch
import numpy as np

def sigmoid_manual(x):
    a = -x
    b = np.exp(a)
    c = 1+b
    z = 1.0/c

    # backward
    dzdc = -(1.0/(c**2))
    dzdb = dzdc * 1
    dzda = dzdb * np.exp(a)
    dzdx = dzda * (-1)

    return dzdx

def sigmoid(x):
    return 1.0/(1.0 + torch.exp(-x))

inp_x = 2.0
x = torch.tensor(2.0, requires_grad=True)

y = sigmoid(x)
y.backward()

print(x.grad)
print(sigmoid_manual(inp_x))
