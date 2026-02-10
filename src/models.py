import torch
import torch.nn as nn
import torch.nn.functional as F


class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28,400)
        self.fc2 = nn.Linear(400,200)
        self.fc3 = nn.Linear(200,100)
        self.fc4 = nn.Linear(100,10)
    
    def forward(self, x):
        x = x.view(-1, 28*28)  # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1,16,3,1)  # (input_channels, output_channels, Kernel_size, stride)
        self.pool = nn.MaxPool2d(2,2)  # (Kernel_size, stride)
        self.conv2 = nn.Conv2d(16,32,3,1)
        self.fc1 = nn.Linear(5*5*32,100)
        self.fc2 = nn.Linear(100,50)
        self.fc3 = nn.Linear(50,10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 5*5*32)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

class CIFAR10_CNN(nn.Module):
    def __init__(self):
        super(CIFAR10_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5,1)  # (input_channels, output_channels, Kernel_size, stride)
        self.pool = nn.MaxPool2d(2,2)  # (Kernel_size, stride)
        self.conv2 = nn.Conv2d(6,16,5,1)
        self.fc1 = nn.Linear(5*5*16,100)
        self.fc2 = nn.Linear(100,50)
        self.fc3 = nn.Linear(50,10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 5*5*16)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x