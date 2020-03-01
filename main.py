import torch
import torchvision

from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# Download MNIST dataset
train = datasets.QMNIST("", train=True, download=True, transform=(transforms.Compose([transforms.ToTensor()])))

test = datasets.QMNIST("", train=False, download=True, transform=(transforms.Compose([transforms.ToTensor()])))

# Load data
train_data = torch.utils.data.DataLoader(train, batch_size=12, shuffle=True)

test_data = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True)

# Feedforward neural network
class FeedforwardNet(nn.Module):
    def __init__(self):
        super(FeedforwardNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 56)
        self.fc2 = nn.Linear(56, 56)
        self.fc3 = nn.Linear(56, 56)
        self.fc4 = nn.Linear(56, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

# Initialise model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = FeedforwardNet().to(device)

# Model training
optimiser = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(30):
    for data in train_data:
        X, y = data
        X, y = X.to(device), y.to(device)
        model.zero_grad()
        output = model(X.view(-1, 28*28))
        loss = F.nll_loss(output, y)
        loss.backward()
        optimiser.step()
    print(loss)

# Test accuracy

correct = 0
total = 0

with torch.no_grad():
    for data in test_data:
        X, y = data
        X, y = X.to(device), y.to(device)
        output = model(X.view(-1,784))
        
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1

print("Accuracy: ", round(correct/total, 3))
