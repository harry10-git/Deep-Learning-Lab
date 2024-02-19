import torch
from matplotlib import pyplot as plt
from torch.autograd import grad

# Create the tensors x and y. They are the training
# examples in the dataset for the linear regression

x = torch.tensor(
    [12.4, 14.3, 14.5, 14.9, 16.1, 16.9, 16.5, 15.4, 17.0, 17.9, 18.8, 20.3, 22.4, 19.4, 15.5, 16.7, 17.3, 18.4, 19.2,
     17.4, 19.5, 19.7, 21.2])
y = torch.tensor(
    [11.2, 12.5, 12.7, 13.1, 14.1, 14.8, 14.4, 13.4, 14.9, 15.6, 16.4, 17.7, 19.6, 16.9, 14.0, 14.6, 15.1, 16.1, 16.8,
     15.2, 17.0, 17.2, 18.6])


# The learning rate is set to alpha = 0.01
learning_rate = torch.tensor(0.001)


class RegressionModel:
    def __init__(self):
        self.w = torch.rand([1], requires_grad=True)
        self.b = torch.rand([1], requires_grad=True)

    def forward(self, x):
        return self.w * x + self.b
    def update(self):
        # Update the weight based on gradient descent
        # equivalently one may write w1.copy_(w1 - learning_rate * w1.grad)
        self.w -= learning_rate * self.w.grad
        self.b -= learning_rate * self.b.grad

    def reset_grad(self):
        self.w.grad.zero_()
        self.b.grad.zero_()




def criterion(yj, y_p):
    return (yj - y_p)**2


model = RegressionModel()

#The list of loss values for the plotting purpose
loss_list = []

# Run the training loop for N epochs
for epochs in range(100):
    loss = 0.0
    #Accumulate the loss for all the samples
    for j in range(len(x)):
        y_p = model(x[j])
        loss +=criterion(y[j], y_p)

    #Find the average loss
    loss = loss / len(x)
    #Add the loss to a list for the plotting purpose
    loss_list.append(loss.item())

    #Compute the gradients using backward dl/dw and dl/db
    loss.backward()

    # Without modifying the gradient in this block
    # perform the operation
    with torch.no_grad():
        model.update()
    #reset the gradients for next epoch
    model.reset_grad()
    #w.grad = None
    #b.grad = None

    # prev_loss = loss
    #Display the parameters and loss
    print("The parameters are w={},  b={}, and loss={}".format(model.w, model.b, loss.item()))

#Display the plot
plt.plot(loss_list)
plt.show()
