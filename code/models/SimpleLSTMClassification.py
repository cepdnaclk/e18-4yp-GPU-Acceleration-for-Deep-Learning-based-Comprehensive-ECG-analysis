import torch.nn as nn


class SimpleLSTMClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=8, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(0.2)

        self.MLP = nn.Sequential(
            nn.Linear(256, 1000),
            nn.ReLU(),
            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Linear(100, 5),
        )

    def forward(self, x):
        batch_size ,seq_len ,num_channels = x.size()
        x = x.view(batch_size, seq_len, num_channels)  # Reshape input for LSTM
        output, _ = self.lstm(x)
        output = output[:, -1, :]  # Get the last hidden state
        output = self.dropout(output)
        output = self.MLP(output)
        return output
