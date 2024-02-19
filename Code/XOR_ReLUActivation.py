import torch
from matplotlib import pyplot as plt
from torch.autograd import grad

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
loss_list = []
torch.manual_seed(42)

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
        loss = loss_fn (outputs.flatten(), labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        total_loss += loss.item()

    return total_loss/(len(train_data_loader) * batch_size)



# Create the tensors x1,x2 and y. They are the training
# examples in the dataset for the XOR
X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
Y = torch.tensor([0,1,1,0], dtype=torch.float32)




class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        #self.w = torch.nn.Parameter(torch.rand([1]))
        #self.b = torch.nn.Parameter(torch.rand([1]))

        self.linear1 = nn.Linear(2,2,bias=True)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(2,1,bias=True)
        #self.activation2 = nn.ReLU()
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        #x = self.activation2(x)
        return x

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


batch_size = 1
# Create the dataloaders for reading data
# This provides a way to read the dataset in batches, also shuffle the data
# also many more
train_data_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
print(len(train_data_loader))
#Find if CUDA is available to load the model and device
# on to the available device CPU/GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model to GPU
model = XORModel().to(device)
print(model)

# add the criterion which is the MSELoss
loss_fn = torch.nn.MSELoss() #BCEWithLogitsLoss()

# Optimizers specified in the torch.optim package
optimizer = torch.optim.SGD(model.parameters(), lr=0.03)



EPOCHS = 20000
#If needed keep track of the validation loss
#and associated parameter values for saving the
#optimal parameter values
best_vloss = 1_000_000. #Inf


for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch)
    loss_list.append(avg_loss)

    print('LOSS train {}'.format(avg_loss))


for param in model.named_parameters():
    print(param)


#Inference
input = torch.tensor([0, 0], dtype=torch.float32).to(device)
#Model inference
model.eval()
print("The input is = {}".format(input))
print("Output y predicted ={}".format(model(input)))
#Display the plot
plt.plot(loss_list)
plt.show()

# #
# # # Function to plot decision boundary
# def plot_decision_boundary(X, y, model, title):
#     h = .02  # Step size in the mesh
#     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx, yy = torch.meshgrid(torch.arange(x_min, x_max, h), torch.arange(y_min, y_max, h)))
#
#     # Concatenate the meshgrid points for prediction
#     Z = model(xx.ravel())
#     Z = Z.reshape(xx.shape)
#
#     plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
#     plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
#     plt.title(title)
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     plt.show()
#
#
# # Plot decision boundary
# plot_decision_boundary(X, Y, model, 'Decision Boundary for XOR Problem')

# model_params = list(model.parameters())
# model_weights = model_params[0].data.numpy()
# model_bias = model_params[1].data.numpy()
#
# plt.scatter(X.numpy()[[0,-1], 0], X.numpy()[[0, -1], 1], s=50)
# plt.scatter(X.numpy()[[1,2], 0], X.numpy()[[1, 2], 1], c='red', s=50)
#
# x_1 = np.arange(-0.1, 1.1, 0.1)
# y_1 = ((x_1 * model_weights[0,0]) + model_bias[0]) / (-model_weights[0,1])
# plt.plot(x_1, y_1)
#
# x_2 = np.arange(-0.1, 1.1, 0.1)
# y_2 = ((x_2 * model_weights[1,0]) + model_bias[1]) / (-model_weights[1,1])
# plt.plot(x_2, y_2)
# plt.legend(["neuron_1", "neuron_2"], loc=8)
# plt.show()