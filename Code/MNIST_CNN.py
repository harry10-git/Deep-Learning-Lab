import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from collections import OrderedDict

import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor

def train_one_epoch(epoch_index):
    total_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_data_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

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




class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        #self.w = torch.nn.Parameter(torch.rand([1]))
        #self.b = torch.nn.Parameter(torch.rand([1]))


        self.net =nn.Sequential(nn.Conv2d(1,64,kernel_size=3),
                                nn.ReLU(),
                                nn.MaxPool2d((2,2), stride=2),
                                nn.Conv2d(64, 128, kernel_size=3),
                                nn.ReLU(),
                                nn.MaxPool2d((2,2), stride=2),
                                nn.Conv2d(128, 64, kernel_size=3),
                                nn.ReLU(),
                                nn.MaxPool2d((2, 2), stride=2)
        )
        self.classification_head = nn.Sequential(nn.Linear(64, 20, bias=True),
                                nn.ReLU(),
                                nn.Linear(20, 10, bias=True))

    def forward(self, x):
        features = self.net(x)
        return self.classification_head(features.view(batch_size,-1))


#Create the dataset
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform = ToTensor())
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform = ToTensor())

batch_size = 4
# Create the dataloaders for reading data
# This provides a way to read the dataset in batches, also shuffle the data
# also many more
train_data_loader = DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(mnist_testset, batch_size=batch_size, shuffle=False)

#Just to display a sample image from data loader
#and the corresponding label
# images, labels = next(iter(train_data_loader))
# plt.imshow(images[0].reshape(28,28), cmap="gray")
# plt.title(labels[0])
# plt.show()

#Find if CUDA is available to load the model and device
# on to the available device CPU/GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model to GPU
model = CNNClassifier().to(device)
print(model)


total_params = 0
for name, param in model.named_parameters():
    print(name)
    params = param.numel()
    print(params)
    total_params += params
print("Total Parameters:{}".format(total_params))


# add the criterion which is the MSELoss
loss_fn = torch.nn.CrossEntropyLoss()

# Optimizers specified in the torch.optim package
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)



EPOCHS = 10


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
    print('LOSS train {}'.format(avg_loss ))

images, true_labels = next(iter(test_loader))
plt.imshow(images[0].reshape(28,28), cmap="gray")
model.eval()

correct = 0
total = 0
for i, vdata in enumerate(test_loader):
    tinputs, tlabels = vdata
    tinputs = tinputs.to(device)
    tlabels = tlabels.to(device)
    toutputs = model(tinputs)
    #Select the predicted class label which has the
    # highest value in the output layer
    _, predicted = torch.max(toutputs, 1)
    print("True label:{}".format(tlabels))
    print('Predicted: {}'.format(predicted))
    # Total number of labels
    total += tlabels.size(0)

    # Total correct predictions
    correct += (predicted == tlabels).sum()

accuracy = 100.0 * correct / total
print("The overall accuracy is {}".format(accuracy))