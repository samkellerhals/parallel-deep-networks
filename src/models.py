import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import time
import json
from utils import find_latest_log

# Feedforward neural network
class FeedforwardNet(nn.Module):
    def __init__(self):
        super(FeedforwardNet, self).__init__()
        self.model_type = 'ff'
        self.fc1 = nn.Linear(28*28, 56)
        self.fc2 = nn.Linear(56, 56)
        self.fc3 = nn.Linear(56, 56)
        self.fc4 = nn.Linear(56, 10)
    
    def forward(self, x):
        # Fully connected layer 1
        x = F.relu(self.fc1(x))
        
        # Fully connected layer 2
        x = F.relu(self.fc2(x))

        # Fully connected layer 3
        x = F.relu(self.fc3(x))

        # Fully connected layer 4
        x = self.fc4(x)

        # Compute probabilities
        probs = F.log_softmax(x, dim=1)
        return probs

# Convolutional neural network
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.model_type = 'conv'
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=10,kernel_size=(3,3),stride=1,padding=1) # output size = ((28-3+2(1))/1)+1
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2)) # output size = 28/2
        self.conv2 = nn.Conv2d(in_channels=10,out_channels=20,kernel_size=(3,3),stride=1,padding=1) # output size = ((14-3)+2(1)/1)+1
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2)) # output size = 14/2
        self.fc1 = nn.Linear(in_features=7*7*20, out_features=10)

    def forward(self, x):
        # Conv layer 1
        x = F.relu(self.conv1(x))

        # Maxpool layer 1
        x = self.maxpool1(x)

        # Conv layer 2
        x = F.relu(self.conv2(x))

        # Maxpool layer 2
        x = self.maxpool2(x)

        x = torch.flatten(x, 1)

        # Fully connected layer 1
        x = self.fc1(x)

        # Compute probabilities
        probs = F.log_softmax(x, dim=1)
        return probs

# Train and test models
def train(epochs, arch, model, device, train_loader):
    #train_data = get_train_data()
    optimiser = optim.SGD(model.parameters(), lr=0.001)
    
    total_time = 0

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        pid = os.getpid()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimiser.zero_grad()
            
            if arch == "conv":
                output = model(data)
            else:
                output = model(data.view(-1, 28*28))

            loss = F.nll_loss(output, target)
            
            loss.backward()
            
            optimiser.step()
        
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        total_time += processing_time

    # Write file with training metrics

        latest_log = find_latest_log()

    with open(latest_log) as f:
        data = json.load(f)
        data['training_time'] = total_time
        new_data = json.dumps(data)
    
    with open(latest_log,'w') as f:
        f.write(new_data)
        f.close()

    print(f'Model metrics have been saved at: {latest_log}')

def test(model, device, test_loader, arch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            if arch == "conv":
                output = model(data)
            else:
                output = model(data.view(-1, 28*28))

            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

     # Write file with training metrics

    latest_log = find_latest_log()

    with open(latest_log) as f:
        data = json.load(f)
        data['accuracy'] = 100. * correct / len(test_loader.dataset)
        data['avg_loss'] = test_loss
        new_data = json.dumps(data)
    
    with open(latest_log,'w') as f:
        f.write(new_data)
        f.close()

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == "__main__":
    pass