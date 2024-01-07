import torch.nn as nn


class SimpleNeuralNetwork(nn.Module):
    def __init__(self):
        super(SimpleNeuralNetwork, self).__init__()
        self.all = nn.Sequential(
            nn.Linear(40000, 20000),
            nn.ReLU(),
            nn.Linear(20000, 10000),
            nn.ReLU(),
            nn.Linear(10000, 5000),
            nn.ReLU(),
            nn.Linear(5000, 2500),
            nn.ReLU(),
            nn.Linear(2500, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )
        # self.all = nn.Sequential(
        #     nn.Linear(40000, 10),
        #     nn.ReLU(),
        #     nn.Linear(10, 1),
        # )

    def forward(self, x):
        x = self.all(x)
        return x
