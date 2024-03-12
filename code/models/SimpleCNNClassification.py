import torch.nn as nn

class SimpleCNNClassification(nn.Module):
    def __init__(self):
        super(SimpleCNNClassification, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(40000, 1000)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(1000, 100)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(100, 10)
        self.relu5 = nn.ReLU()
        self.fc4 = nn.Linear(10, 5)

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        # Flatten before fully connected layers
        x = x.view(-1, 40000)

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        x = self.relu5(x)
        x = self.fc4(x)

        return x
