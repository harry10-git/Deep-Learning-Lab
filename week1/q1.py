# reshaping , viewing, stacking, squeezing and unsqueezing

import torch

a = torch.tensor([1,2,3,4,5,6])
print('before reshape : ',a)

a = a.reshape([3,2])
print('after reshape : ',a)

b = torch.tensor([1,2,3,4,5,6,7,8,9])
print('view: ' ,b.view(3,3))

c1= torch.tensor([1,2,3])
c2 = torch.tensor([4,5,6])
c = torch.stack((c1,c2), dim=0)
print('stack (dim 0): \n' ,c)

c = torch.stack((c1,c2), dim=1)
print('stack (dim 1): \n' ,c)


d1 = torch.randn(3,1,2,1,5)
d = torch.squeeze(d1)
print('d.size() = ',d.size())

e1 = torch.arange(8, dtype=torch.float)
print('e1 = ', e1)

e = torch.unsqueeze(e1, dim=0)
print('unsqueeze dim =0 :  ', e.size())

e = torch.unsqueeze(e1, dim=1)
print('unsqueeze dim =1 :  ', e.size())