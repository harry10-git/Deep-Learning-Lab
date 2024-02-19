import torch
#Z=3x2+5xy- compute
x = torch.tensor([2], requires_grad=True, dtype=torch.float32)
#y = torch.tensor([3], requires_grad=True, dtype=torch.float32)

a = x**2 + 2*x
a.retain_grad()
# b = 3*a
# b.retain_grad()
# c = x*y
# d = 5*c
# e = b + d
# z = torch.log(e)
z = a
for i in range(3):
    z.backward(retain_graph=True)
print(a.grad)


