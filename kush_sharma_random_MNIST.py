import torch  # Import the PyTorch library for deep learning
import torch.nn as nn  # Import the neural network module from PyTorch
import torch.optim as optim  # Import the optimization module from PyTorch
from torch.utils.data import DataLoader  # Import the data loader module from PyTorch
from torchvision import datasets, transforms  # Import datasets and transforms from the torchvision library


# Define a simple neural network model
class NeuralNetwork(nn.Module):  # Define a custom neural network class that extends the nn.Module
    # class
    def __init__(self):  # Define the initialization method for the neural network class
        super(NeuralNetwork, self).__init__()  # Call the initialization method of the nn.Module class
        self.fc1 = nn.Linear(784, 256)  # Create a fully connected layer with 784 input features and 256
        # output features
        self.fc2 = nn.Linear(256, 10)  # Create a fully connected layer with 256 input features and 10
        # output features

    def forward(self, x):  # Define the forward method to specify how data flows through the
        # network
        x = x.view(x.size(0), -1)  # Reshape the input tensor to have a shape of (batch_size, 784)
        # Apply the first fully connected layer followed by a ReLU activation function
        x = torch.relu(self.fc1(x))
        # Apply the second fully connected layer
        x = self.fc2(x)
        return x  # Return the output tensor


# Function to train the neural network
def train(model, train_loader, criterion, optimizer):
    model.train()  # Set the model to training mode
    for data, target in train_loader:  # Iterate over the training dataset
        optimizer.zero_grad()  # Zero the gradients of the model parameters
        output = model(data)  # Perform a forward pass of the data through the model
        loss = criterion(output, target)  # Compute the loss between the model's output and the target labels
        loss.backward()  # Perform backpropagation to compute the gradients
        optimizer.step()  # Update the model parameters using the computed gradients


# Function to evaluate the neural network
def evaluate(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0  # Initialize a counter for the number of correct predictions
    total = 0  # Initialize a counter for the total number of predictions
    with torch.no_grad():  # Disable gradient computation during evaluation
        for data, target in test_loader:  # Iterate over the test dataset
            output = model(data)  # Perform a forward pass of the data through the model
            _, predicted = torch.max(output.data, 1)  # Get the index of the predicted class with the highest
            # probability
            total += target.size(0)  # Increment the total counter by the batch size
            correct += (predicted == target).sum().item()  # Count the number of correct predictions
    accuracy = 100 * correct / total  # Compute the accuracy as a percentage
    return accuracy  # Return the accuracy


# Define data transformations
transform = transforms.Compose([  # Define a sequence of data transformations
    transforms.ToTensor(),  # Convert the image data to tensors
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize the image data with mean and standard deviation values
])

# Load MNIST dataset
train_dataset = datasets.MNIST(root='data', train=True, download=True,
                               transform=transform)  # Load the MNIST training dataset
test_dataset = datasets.MNIST(root='data', train=False, download=True,
                              transform=transform)  # Load the MNIST test dataset

# Set initial labeled dataset size
initial_labeled_size = 100

# Create initial labeled and unlabeled datasets
labeled_dataset = \
torch.utils.data.random_split(train_dataset, [initial_labeled_size, len(train_dataset) - initial_labeled_size])[0]
unlabeled_dataset = \
torch.utils.data.random_split(train_dataset, [initial_labeled_size, len(train_dataset) - initial_labeled_size])[1]

# Define the neural network model
model = NeuralNetwork()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Define the cross-entropy loss function
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Define the stochastic gradient descent optimizer

# Train the initial model
train_loader = DataLoader(labeled_dataset, batch_size=64, shuffle=True)  # Create a data loader for the labeled dataset
train(model, train_loader, criterion, optimizer)  # Train the model using the labeled dataset

# Evaluate the initial model
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)  # Create a data loader for the test dataset
accuracy = evaluate(model, test_loader)  # Evaluate the model on the test dataset
print("Initial Model Accuracy:", accuracy)  # Print the accuracy of the initial model

# Active Open-Set Annotation using the Random Method
num_iterations = 5
additional_labeled_size = 20

for iteration in range(num_iterations):  # Iterate for the specified number of iterations
    additional_labeled_dataset, unlabeled_dataset = torch.utils.data.random_split(unlabeled_dataset,
                                                                                  [additional_labeled_size,
                                                                                   len(unlabeled_dataset) - additional_labeled_size])
    # Randomly select additional instances to label
    labeled_dataset = torch.utils.data.ConcatDataset([labeled_dataset, additional_labeled_dataset])
    # Add newly labeled instances to the labeled dataset

    train_loader = DataLoader(labeled_dataset, batch_size=64,
                              shuffle=True)  # Create a data loader for the updated labeled dataset
    train(model, train_loader, criterion, optimizer)  # Train the model with the updated labeled dataset

    accuracy = evaluate(model, test_loader)  # Evaluate the model on the test dataset
    print("Iteration:", iteration + 1, "Accuracy:", accuracy)  # Print the accuracy after each iteration
