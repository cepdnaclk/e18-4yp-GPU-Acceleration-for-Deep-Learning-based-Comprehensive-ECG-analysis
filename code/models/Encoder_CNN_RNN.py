import torch
import torch.nn as nn
import math

class TransformerEncoderModel(nn.Module):
    def __init__(self, input_size, patch_size, num_layers, num_heads, dim_feedforward, output_size, dropout=0.1):
        super(TransformerEncoderModel, self).__init__()
        self.embedding_layer = nn.Linear(patch_size, input_size)
        self.pos_encoder = PositionalEncoding(input_size, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.patch_size = patch_size

    def forward(self, x):
        batch_size, _ = x.size()
        num_patches = x.size(1) // self.patch_size
        x = x.view(batch_size, num_patches, self.patch_size)

        x = self.embedding_layer(x)
        x = self.pos_encoder(x)

        x = self.encoder(x)
        x = self.dropout(x)

        x = x.mean(dim=1)

        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class CNN_RNN_Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(CNN_RNN_Model, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.rnn = nn.GRU(128, 64, batch_first=True)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size, seq_len = x.shape
        x = x.view(batch_size, 1, seq_len)  # Add channel dimension
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.transpose(1, 2)  # Swap dimensions for RNN
        _, x = self.rnn(x)  # Use the final hidden state
        x = x.squeeze(0)  # Remove the sequence dimension
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x