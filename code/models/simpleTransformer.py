import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                dropout=dropout
            ),
            num_layers=num_layers
        )
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # Rearrange dimensions for the transformer
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # Take the mean across the sequence dimension
        x = self.linear(x)
        return x