import torch

x = torch.tensor(3.5, requires_grad=True)

y = x*x
z = 2*y + 3

z.backward()

print(x.grad)
