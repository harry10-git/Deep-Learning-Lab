import torch

random_tensor1 = torch.rand(7, 7)

random_tensor2 = torch.rand(1, 7)

# tranpose 2
random_tensor2 = torch.transpose(random_tensor2, 0, 1)
out = torch.matmul(random_tensor1, random_tensor2)

print('out: ' ,out)

max_element = torch.argmax(out)
min_element = torch.argmin(out)
print('max index: ' ,max_element)
print('min index: ' ,min_element)