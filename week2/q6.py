import torch
import numpy as np

x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(1.0, requires_grad=True)
z = torch.tensor(1.0, requires_grad=True)

f = torch.tanh_(torch.log_(1 + z*(2.0*x/torch.sin(y))))
f.backward()

# manual
def manual(x,y,z):
    a = 2 * x
    b = np.sin(y)
    c = a/b
    d = c*z
    e = np.log(d+1)
    f = np.tanh(e)

    dadx = 2
    dcda = 1/b
    dddc = z
    dedd = 1/(d+1)
    dfde = 1 - np.tanh(e)**2

    return dfde * dedd * dddc * dcda * dadx


print('backward: ')
print(x.grad)
'''print(y.grad)
print(z.grad)'''

print('manual')
print(manual(1,1,1))

