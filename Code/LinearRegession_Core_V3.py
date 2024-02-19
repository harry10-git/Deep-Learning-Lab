import torch
from matplotlib import pyplot as plt
from torch.autograd import grad

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

def train_one_epoch(epoch_index):
    total_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_data_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        total_loss += loss.item()

    return total_loss/(len(train_data_loader) * batch_size)



# Create the tensors x and y. They are the training
# examples in the dataset for the linear regression
# X = torch.tensor(
#     [12.4, 14.3, 14.5, 14.9, 16.1, 16.9, 16.5, 15.4, 17.0, 17.9, 18.8, 20.3, 22.4, 19.4, 15.5, 16.7, 17.3, 18.4, 19.2,
#      17.4, 19.5, 19.7, 21.2])
# Y = torch.tensor(
#     [11.2, 12.5, 12.7, 13.1, 14.1, 14.8, 14.4, 13.4, 14.9, 15.6, 16.4, 17.7, 19.6, 16.9, 14.0, 14.6, 15.1, 16.1, 16.8,
#      15.2, 17.0, 17.2, 18.6])

X = torch.tensor([5,7,12, 16,20], dtype=torch.float32)
Y = torch.tensor([40,120,180,210,240],dtype=torch.float32)

# The learning rate is set to alpha = 0.003
learning_rate = torch.tensor(0.003)

class RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.rand([1]))
        self.b = torch.nn.Parameter(torch.rand([1]))

    def forward(self, x):
        return self.w * x + self.b


class MyDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].to(device), self.Y[idx].to(device)



#Create the dataset
full_dataset = MyDataset(X, Y)

#Split the dataset into training set and validation set
train_size = int(0.7 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, validation_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

batch_size = 1
# Create the dataloaders for reading data
# This provides a way to read the dataset in batches, also shuffle the data
# also many more
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

#Find if CUDA is available to load the model and device on to the available device CPU/GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model to GPU
model = RegressionModel().to(device)
print(model)

# add the criterion which is the MSELoss
loss_fn = torch.nn.MSELoss()

# Optimizers specified in the torch.optim package
optimizer = torch.optim.SGD(model.parameters(), lr=0.003)



EPOCHS = 200
#If needed keep track of the validation loss
#and associated parameter values for saving the
#optimal parameter values
best_vloss = 1_000_000. #Inf


for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch)

    running_vloss = 0.0

    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss
    avg_vloss = running_vloss / len(validation_loader)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        #model_path = 'model_{}'.format(epoch + 1)
        #torch.save(model.state_dict(), model_path)

for param in model.named_parameters():
    print(param)
# plot
plt.figure(figsize=(4, 3))
ax = plt.axes()
ax.scatter(X, Y)

#Verify the estimated parameters
#by plotting the graph using the parameters
xtest = torch.linspace(10,30, steps=10).to(device)
ypred = model.w * xtest   + model.b
ax.plot(xtest.detach().cpu(), ypred.detach().cpu(), color='red')
ax.set(xlim=(12, 20), xticks=np.arange(12, 20),
       ylim=(10, 24), yticks=np.arange(10, 24))
plt.show()

