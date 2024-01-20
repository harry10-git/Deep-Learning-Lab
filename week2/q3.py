import torch
import numpy as np

x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

u = w*x
v = u+b

a = 1.0/(1.0 + torch.exp(v))

a.backward()

# print(x.grad)
print(w.grad)
print(b.grad)

def sigmoid_manual(x):
    a = -x
    b = np.exp(a)
    c = 1+b
    z = 1.0/c
    dzdc = -(1.0/(c**2))
    dzdb = dzdc * 1
    dzda = dzdb * np.exp(a)
    dzdx = dzda * -1
    return dzdx

def manual(x,w, b):
    u = x*w
    v = u+b
    a = 1.0/(1 + np.exp(v))
    dadv = sigmoid_manual(v)
    dadu = dadv * 1
    dadb = dadv * 1
    dadw = dadu * x
    return dadw,dadb

print(manual(2,3,1))