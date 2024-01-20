import torch

def manual(b,x,w):
    u = w*x
    v = u+b
    a = max(0.0,v)

    if a == 0:
        return 0.0
    else:
        dadv = 1
        dadu = 1
        dadw = x

        return dadw


print('manual')
print(manual(1,2,3))

b = torch.tensor(1.0, requires_grad=True)
x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)

u = w*x
v = u+b
a = torch.nn.ReLU()
y = a(v)
y.backward()

print('function')
print(w.grad)