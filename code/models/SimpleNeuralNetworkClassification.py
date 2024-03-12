import torch.nn as nn


class SimpleNeuralNetworkClassification(nn.Module):
    def __init__(self):
        super(SimpleNeuralNetworkClassification, self).__init__()
        self.all = nn.Sequential(
            nn.Linear(40000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
        )

    def forward(self, x):
        x = self.all(x)
        return x


