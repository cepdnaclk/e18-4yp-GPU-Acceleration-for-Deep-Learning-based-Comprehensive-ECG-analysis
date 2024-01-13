import torch.nn as nn


class SimpleLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=40000, hidden_size=100, num_layers=1, batch_first=True
        )
        self.MLP = nn.Sequential(
            nn.Linear(100, 1000),
            nn.ReLU(),
            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.MLP(x)
        return x
