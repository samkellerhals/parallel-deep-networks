from data import *

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

# Convolutional neural network


# Train convolutional neural net



# Train feedforward neural net
class train_feedforward():
    def __init__(self, epochs):
        self.epochs = epochs
        self.model = FeedforwardNet().to(device)
        self.optimiser = optim.SGD(self.model.parameters(), lr=0.001)
    
    def train(self):
        for epoch in range(self.epochs):
            for data in train_data:
                X, y = data
                X, y = X.to(device), y.to(device)
                self.model.zero_grad()
                output = self.model(X.view(-1, 28*28))
                loss = F.nll_loss(output, y)
                loss.backward()
                self.optimiser.step()
            print(loss)

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