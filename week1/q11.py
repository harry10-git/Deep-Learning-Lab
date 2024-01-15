import torch

torch.manual_seed(7)
random_tensor = torch.randn(1, 1, 1, 10)

new_tensor = random_tensor.squeeze()

# Print the first tensor and its shape
print("Original Tensor:")
print(random_tensor)



print("New Tensor:")
print(new_tensor)
