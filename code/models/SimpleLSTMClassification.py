import torch.nn as nn


class SimpleLSTMClassification(nn.Module):
    def __init__(self, input_size=40000):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=5000, hidden_size=100, num_layers=1, batch_first=True
        )
        self.MLP = nn.Sequential(
            nn.Linear(100, 1000),
            nn.ReLU(),
            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Linear(100, 5),
        )
        
        self.softmax = nn.Softmax()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Get the last hidden state
        x = self.MLP(x)
        x = self.softmax(x)
        return x