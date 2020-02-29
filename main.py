import torch
import torchvision

from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F

# Download MNIST dataset
train = datasets.QMNIST("", train=True, download=True, transform=(transforms.Compose(
    [transforms.CenterCrop(10),
    transforms.ToTensor()
    ])))

test = datasets.QMNIST("", train=False, download=True, transform=(transforms.Compose([
    transforms.ToTensor()])))

# Feedforward neural network
class FeedforwardNet(nn.Module):
    def __init__(self):
        super(FeedforwardNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 56)
        self.fc2 = nn.Linear(56, 56)
        self.fc3 = nn.Linear(56, 56)
        self.output = nn.Linear(56, 9)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.output(x))
        return x

# Initialise model
model = FeedforwardNet()

# Create test data
test_data = torch.Tensor(28,28).view(28*28)
test_pass = model.forward(test_data)




        


        
        
