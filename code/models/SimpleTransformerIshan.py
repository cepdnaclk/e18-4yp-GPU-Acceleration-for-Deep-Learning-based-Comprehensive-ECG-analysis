import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, output_size):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden_size, num_heads), num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # Reshape to (seq_length, batch_size, hidden_size)
        x = self.transformer(x)
        x = x.mean(dim=0)  # Average over the sequence length
        x = self.fc(x)
        return x


