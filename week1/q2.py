# use of permute
# import pytorch library
import torch

# creating a tensor with random
# values of dimension 3 X 5 X 2
input_var = torch.randn(3, 5, 2)

# print size
print(input_var.size())

print(input_var)

# dimensions permuted
input_var = input_var.permute(2, 0, 1)

# print size
print(input_var.size())

print(input_var)
