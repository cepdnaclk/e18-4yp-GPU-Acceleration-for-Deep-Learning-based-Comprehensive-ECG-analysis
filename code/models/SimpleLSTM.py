import torch.nn as nn


class SimpleLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=40000, hidden_size=500, num_layers=1, batch_first=True
        )
        self.MLP1 = nn.Sequential(
            nn.Linear(500, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1),
        )

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.MLP1(x)
        return x
