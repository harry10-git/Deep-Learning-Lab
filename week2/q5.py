import torch

x = torch.tensor(2.0, requires_grad=True)
f = (8 * (x**4)) + (3 * (x**3)) + (7 * (x**2)) + (6*x) +3
f.backward()
print(x.grad)


def manual(x):
    f = (8 * (x**4)) + (3 * (x**3)) + (7 * (x**2)) + (6*x) +3
    dfdx = 32 * x**3 + 9*x**2 + 14*x + 6

    return dfdx

print('manual')
print(manual(2))