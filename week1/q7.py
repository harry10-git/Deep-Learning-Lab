# generate 2 random tensors of size 3X2 and send them to gpu

import torch

a = torch.rand(2,3)
b = torch.rand(2,3)

print('a.device = ',a.device)
print('b.device = ',b.device)

# send to gpu
print(torch.cuda.is_available())

if torch.cuda.is_available():
   a = a.cuda()
   b = b.cuda()

print('a.device = ',a.device)
print('b.device = ',b.device)