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








#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# import torch
# import torch.nn as nn
# import math

# class TransformerEncoderModel(nn.Module):
#     def __init__(self, input_size, patch_size, num_layers, num_heads, dim_feedforward, output_size):
#         super(TransformerEncoderModel, self).__init__()
#         self.embedding_layer = nn.Linear(patch_size, input_size)
#         self.pos_encoder = PositionalEncoding(input_size)
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads, dim_feedforward=dim_feedforward)
#         self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
#         # Output layer: Deep CNN model
#         self.cnn_output = nn.Sequential(
#             nn.Conv1d(input_size, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2),
#             nn.Conv1d(128, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2),
#             nn.Conv1d(64, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2),
#             nn.Flatten(start_dim=2),
#             nn.Linear(32 * (input_size // 8), output_size)
#         )
        
#         self.patch_size = patch_size

#     def forward(self, x):
#         batch_size, _ = x.size()
#         num_patches = x.size(1) // self.patch_size
#         x = x.view(batch_size, num_patches, self.patch_size)
#         x = self.embedding_layer(x)
#         x = self.pos_encoder(x)
#         x = self.encoder(x)
#         print("x.shape : ", x.shape)
#         x = x.permute(1, 2, 0)  # Permute to (sequence_length, input_size, batch_size)
#         x = self.cnn_output(x)
#         return x

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(0.1)
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)