import torch
import torchvision

# You can modify the following parameters to see how the performance of the model changed
"""This line sets the number of training epochs to 3. An epoch is a complete pass through the entire training 
dataset. The higher the number of epochs, the more the network learns about the data. However, after a certain point, 
the network might start over-fitting, which means it performs well on the training data but poorly on unseen data (
like validation or test sets). """
n_epochs = 3
"""This line sets the batch size for the training data to 64. Batch size is the number of training examples used in 
one iteration, i.e., one forward and backward pass of the neural network. """
batch_size_train = 64
"""This line sets the batch size for the testing data to 1000. This means when you're evaluating the model on your 
test data, you're looking at 1000 examples at once. """
batch_size_test = 1000
"""This line sets the learning rate for the training algorithm (like Gradient Descent) to 0.01. The learning rate 
controls how much to change the model in response to the estimated error each time the model weights are updated. """
learning_rate = 0.01
"""This line sets the momentum for the optimizer to 0.5. Momentum is a term used in the context of optimizers (
especially SGD, or Stochastic Gradient Descent) that helps accelerate the optimizer in the right direction and 
dampens oscillations. """
momentum = 0.5
"""This line sets the logging interval to 10. This means that for every 10 batches of the training data, 
some operation (like printing the current state of training or saving the model) is performed. In your previous code, 
after every 10 batches, the model's and optimizer's states were saved. """
log_interval = 10

random_seed = 1
#  disables cuDNN, a GPU-accelerated library for deep neural networks
torch.backends.cudnn.enabled = False
"""This sets the seed for generating random numbers for PyTorch's random number generator. Again, this is done for 
consistency in results across multiple executions of the code. Note that this does not guarantee complete 
reproducibility in results due to other factors (like multithreading in CPUs and some non-deterministic operations in 
GPUs). For complete reproducibility, more steps might be needed, including setting the seed for NumPy's random number 
generator (if NumPy is used), and possibly disabling multithreading. """
torch.manual_seed(random_seed)

train_loader = torch.utils.data.DataLoader(
    # downloads the MNIST dataset (a dataset of handwritten digits) and loads it from the './data/' directory. If the
    # dataset isn't already there, it downloads the data. The train=True argument indicates that it is loading the
    # training data. For the test loader, train=False is used to load the testing data.
    torchvision.datasets.MNIST('./data/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   # convert PIL Images or numpy.ndarray into PyTorch tensors
                                   torchvision.transforms.ToTensor(),
                                   # normalizes the tensor image with the mean and standard deviation provided
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)
"""This creates an enumerate object from the test_loader. An enumerate object adds a counter to an iterable and 
returns it as an enumerate object. Here, examples is an iterable over the batches of the test_loader. Each item from 
the examples iterator is a tuple, where the first element is the index of the batch, and the second element is the 
data for that batch. """
examples = enumerate(test_loader)

"""
The next() function retrieves the next item from the iterator examples. 
"""
# the example_data will be a tensor which is the result of an image been processed by transform
# the example_targets is the true label of the current image tensor
batch_idx, (example_data, example_targets) = next(examples)

print(example_data.shape)

import matplotlib.pyplot as plt

fig = plt.figure()
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
fig.show()

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# `class Net(nn.Module):` - This line is declaring a new class `Net` that inherits from PyTorch's base class for all
# neural network modules, `nn.Module`. By inheritance, we mean that our `Net` class has all the properties of
# `nn.Module`, but can also include additional properties (like the layers defined in `__init__`).
class Net(nn.Module):
    # This is the constructor of the `Net` class. When an object of this class is created, this method will be
    # automatically called.
    def __init__(self):
        # calls the constructor of the parent class `nn.Module`.
        super(Net, self).__init__()
        # This defines the first convolutional layer. The arguments are the number of input channels (1,
        # because images in the MNIST dataset are grayscale), the number of output channels (10), and the kernel size
        # (5x5).
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    # x can be considered as input image and the forward method is a special function that defines the way we
    # compute our output using the given layers and functions.
    def forward(self, x):

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

# creating an instance of the Net class
network = Net()
# creating an instance of the stochastic gradient descent (SGD) optimizer.
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

import os

"""The `train` function is responsible for performing one epoch (one pass through the entire training dataset) of 
training for your network. Here is what each line of the function is doing: 

1. `network.train()`: This sets your model into training mode. Some layers, like Dropout and BatchNorm, 
behave differently during training and evaluation, so this line tells those layers that we are now in training mode. 

2. `for batch_idx, (data, target) in enumerate(train_loader):`: This is the start of a loop that goes over all 
batches of images in the `train_loader`. `data` and `target` are the images and labels for each batch, respectively. 

3. `optimizer.zero_grad()`: Before calculating the gradients and updating the weights, we need to set the existing 
gradients to zero. This is because PyTorch accumulates gradients by default, i.e., the gradient calculated for each 
parameter during backpropagation is added to any previous gradient. 

4. `output = network(data)`: This line passes the input data through the network (i.e., performs a forward pass) and 
stores the output in `output`. 

5. `loss = F.nll_loss(output, target)`: This line calculates the negative log likelihood loss between the network's 
output and the actual target. 

6. `loss.backward()`: This performs a backward pass of the network. In other words, it calculates the gradient of the 
loss with respect to each of the network's parameters. 

7. `optimizer.step()`: This updates the network's parameters using the gradients computed in the backward pass.

8. `if batch_idx % log_interval == 0:`: This checks if the current batch index is a multiple of `log_interval`. If it 
is, the code inside the if-statement is executed. 

9. Inside the if-statement, information about the current epoch, the number of training examples seen so far, 
the total size of the training set, the loss, etc. is printed. The current loss is also added to `train_losses`, 
and the current counter (number of training examples seen so far) is added to `train_counter`. 

10. The `directory` variable is set to the path where you want to save the models. If the directory does not exist, 
it is created with `os.makedirs(directory)`. 

11. The current state of the network and the optimizer (i.e., the values of all parameters and hyperparameters) are 
saved to disk with `torch.save()`. This allows you to load the model later and continue training from where you left 
off. 

In summary, this function trains the model for one epoch, periodically prints information about the training 
progress, saves the training loss and counter, and saves the current state of the model and optimizer. """
def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))

            # define the directory where you want to save the models
            directory = './results'
            # if the directory does not exist
            if not os.path.exists(directory):
                # create the directory
                os.makedirs(directory)

            torch.save(network.state_dict(), directory + '/model.pth')
            torch.save(optimizer.state_dict(), directory + '/optimizer.pth')


def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# test function is called before the training loop starts in order to establish a baseline. This gives you an idea of how
# well your model performs on the test set before any training has occurred. This is useful because it allows you to
# see how much your model improves as a result of the training.
test()
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()

fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
fig.show()

with torch.no_grad():
    output = network(example_data)
# But before that let's again look at a few examples as we did earlier and compare the model's output.
fig = plt.figure()
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Prediction: {}".format(
        output.data.max(1, keepdim=True)[1][i].item()))
    plt.xticks([])
    plt.yticks([])
fig.show()

# Continued Training from Checkpoints

continued_network = Net()
continued_optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                                momentum=momentum)

network_state_dict = torch.load('./results/model.pth')
continued_network.load_state_dict(network_state_dict)

optimizer_state_dict = torch.load('./results/optimizer.pth')
continued_optimizer.load_state_dict(optimizer_state_dict)

for i in range(4, 9):
    test_counter.append(i * len(train_loader.dataset))
    train(i)
    test()

fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
fig.show()
