# numpy array to tensor and back to numpy array
import torch
import numpy as np

npx = np.array([[1,2,3],[5,6,6], [7,8,9]])
print(npx, type(npx))

# change to tensor
t = torch.from_numpy(npx)

print(t, type(t))

npy = t.numpy()

print(npy, type(npy))