import torch

a = torch.tensor(2.0,requires_grad = True)
b = torch.tensor(2.0,requires_grad = True)

x = 2*a + 3*b
y = 5*a*a + 3*b*b*b
z = 2*x + 3*y

z.backward()

def manual(a,b):
    x = 2 * a + 3 * b
    y = 5 * a * a + 3 * b * b * b
    z = 2 * x + 3 * y

    dzda = 2*2 + 3*5*2*a
    dzdb = 2*3 + 3*3*3*(b**2)

    return [dzda, dzdb]


print('function:')
print(a.grad)
print(b.grad)

print('manual: ')
print(manual(2.0, 2.0))
