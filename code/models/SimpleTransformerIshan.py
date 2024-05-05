import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


# class TransformerModel(nn.Module):
#     def __init__(self, d_model=64, nhead=8, num_layers=6, max_len=5000, dropout=0.1):
#         super(TransformerModel, self).__init__()
#         self.pos_encoder = PositionalEncoding(d_model, max_len)
#         encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=dropout)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
#         self.output_layer = nn.Linear(d_model, 1)

#     def forward(self, src):
#         src = self.pos_encoder(src)
#         output = self.transformer_encoder(src)
#         output = self.output_layer(output.mean(dim=1))
#         return output


class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, output_size):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden_size, num_heads), num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        x = self.embedding(x)
        # 32,8,5000 -> 32,5000,8
        x = x.permute(1, 0, 2)  # Reshape to (seq_length, batch_size, hidden_size)
        x = self.transformer(x)
        x = x.mean(dim=0)  # Average over the sequence length
        x = self.fc(x)
        return x


if __name__ == "__main__":
    # Example usage
    batch_size = 8
    input_size = 5000
    d_model = 64

    # Generate random input data
    input_data = torch.randn(batch_size, input_size).unsqueeze(0)
    print(input_data.shape)  # Expected output: torch.Size([8, 5000]
    model = TransformerModel()
    output = model(input_data)
    print(output.shape)  # Expected output: torch.Size([8, 1])
    print(output)
