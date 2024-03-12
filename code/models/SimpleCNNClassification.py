import torch.nn as nn
import torch.nn.functional as F


class SimpleCNNClassification(nn.Module):
    def __init__(self):
        super(SimpleCNNClassification, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(16, 1, kernel_size=3, stride=1, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(40000, 128)  
        self.fc2 = nn.Linear(128, 5)  

    def forward(self, x):
        # Apply convolutional layers with activation functions
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Flatten the tensor for fully connected layers
        x = x.view(x.size(0), -1)

        # Apply fully connected layers with activation functions
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
