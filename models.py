from data import *

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
class init_net():
    def __init__(self, num_epochs, model):
        self.num_epochs = num_epochs
        self.model = model.to(device)
        self.optimiser = optim.SGD(self.model.parameters(), lr=0.001)
    
    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            
            for batch_idx, (data, target) in enumerate(train_data):
                data, target = data.to(device), target.to(device)
                self.optimiser.zero_grad()
                
                if self.model.model_type == "conv":
                    output = self.model(data)
                else:
                    output = self.model(data.view(-1, 28*28))

                loss = F.nll_loss(output, target)
                loss.backward()
                self.optimiser.step()

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_data.dataset), 100. * batch_idx / len(train_data), loss.item()))

    #TODO: write better testing function here.
    def test(self):
        correct = 0
        total = 0

        with torch.no_grad():
            for data in test_data:
                X, y = data
                X, y = X.to(device), y.to(device)
                output = self.model(X.view(-1,784))
                
                for idx, i in enumerate(output):
                    if torch.argmax(i) == y[idx]:
                        correct += 1
                    total += 1
                
            print("Accuracy: ", round(correct/total, 3))


if __name__ == "__main__":
    pass