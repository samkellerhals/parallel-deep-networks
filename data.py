import torch
import torchvision

from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# Setup device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Download MNIST dataset
train = datasets.QMNIST("", train=True, download=True, transform=(transforms.Compose([transforms.ToTensor()])))

test = datasets.QMNIST("", train=False, download=True, transform=(transforms.Compose([transforms.ToTensor()])))

# Load data
train_data = torch.utils.data.DataLoader(train, batch_size=12, shuffle=True)

test_data = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True)