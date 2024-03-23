import torch
import torch.nn as nn
import math

class TransformerEncoderModel(nn.Module):
    def __init__(self, input_size, patch_size, num_layers, num_heads, dim_feedforward, output_size):
        super(TransformerEncoderModel, self).__init__()
        self.embedding_layer = nn.Linear(patch_size, input_size)
        self.pos_encoder = PositionalEncoding(input_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads, dim_feedforward=dim_feedforward)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(input_size, output_size)
        self.patch_size = patch_size  # Store patch_size as a class attribute

    
    def forward(self, x):
        # Reshape input to (batch_size, num_patches, patch_size)
        batch_size, _ = x.size()
        num_patches = x.size(1) // self.patch_size  # Use self.patch_size
        x = x.view(batch_size, num_patches, self.patch_size)    

    # ... (rest of the code)

        # Embed and add positional encoding
        x = self.embedding_layer(x)
        x = self.pos_encoder(x)

        # Pass through the encoder
        x = self.encoder(x)

        # Aggregate the outputs (e.g., mean pooling)
        x = x.mean(dim=1)

        # Pass through the output layer
        x = self.output_layer(x)

        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(0.1)

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