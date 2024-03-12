# import torch.nn as nn


# class SimpleNeuralNetworkClassification(nn.Module):
#     def __init__(self):
#         super(SimpleNeuralNetworkClassification, self).__init__()
#         self.all = nn.Sequential(
#             nn.Linear(40000, 1000),
#             nn.ReLU(),
#             nn.Linear(1000, 100),
#             nn.ReLU(),
#             nn.Linear(100, 10),
#             nn.ReLU(),
#             nn.Linear(10, 5),
#         )

#     def forward(self, x):
#         x = self.all(x)
#         return x


import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(4000, 100, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(100, 10, kernel_size=5)
        self.fc1 = nn.Linear(10, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 10)
        x = self.fc1(x)
        return x
