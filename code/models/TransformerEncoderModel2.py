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
        
#         # Deeper linear output layer
#         self.output_layer = nn.Sequential(
#             nn.Linear(input_size, 512),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(128, output_size)
#         )
        
#         self.patch_size = patch_size

#     def forward(self, x):
#         batch_size, _ = x.size()
#         num_patches = x.size(1) // self.patch_size
#         x = x.view(batch_size, num_patches, self.patch_size)
#         x = self.embedding_layer(x)
#         x = self.pos_encoder(x)
#         x = self.encoder(x)
#         x = x.mean(dim=1)
#         x = self.output_layer(x)
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


# import torch
# import torch.nn as nn
# import math

# class TransformerEncoderModel(nn.Module):
#     def __init__(self, input_size, patch_size, num_layers, num_heads, dim_feedforward, output_size):
#         super(TransformerEncoderModel, self).__init__()
#         self.embedding_layer = nn.Linear(patch_size, input_size)
#         self.pos_encoder = PositionalEncoding(input_size)

#         encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads, dim_feedforward=dim_feedforward)
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

#         # Deeper linear output layer
#         self.output_layer = nn.Sequential(
#             nn.Linear(input_size, 1024),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(256, output_size)
#         )
#         self.patch_size = patch_size

#     def forward(self, x):
#         batch_size, _ = x.size()
#         num_patches = x.size(1) // self.patch_size
#         x = x.view(batch_size, num_patches, self.patch_size)
#         x = self.embedding_layer(x)
#         x = self.pos_encoder(x)
#         x = self.encoder(x)
#         x = x.mean(dim=1)
#         x = self.output_layer(x)
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


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(16, 1, kernel_size=3, stride=1, padding=1)

        # Flattening not required as the Transformer will handle variable sequence lengths

    def forward(self, x):
        # Apply convolutional layers with activation functions
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

class TransformerEncoderModel(nn.Module):
    def __init__(self, input_size, num_layers, num_heads, dim_feedforward, output_size):
        super(TransformerEncoderModel, self).__init__()
        self.embedding_layer = nn.Linear(input_size, input_size)  # Adjust embedding layer
        self.pos_encoder = PositionalEncoding(input_size)

        encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads, dim_feedforward=dim_feedforward)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_size)
        )

    def forward(self, x):
        print('----------------------------------------x.shape : ',x.shape)
        # quit()
        x = self.embedding_layer(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.output_layer(x)
        return x

class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()
        self.cnn = SimpleCNN()
        self.transformer = TransformerEncoderModel(input_size=40000, num_layers=3, num_heads=4, dim_feedforward=512, output_size=1)

    def forward(self, x):
        # Process through CNN
        x = self.cnn(x)

        # Flatten CNN output for Transformer Encoder
        batch_size, channels, seq_length = x.shape
        x = x.view(batch_size, seq_length, channels)

        # Process through Transformer Encoder
        x = self.transformer(x)
        return x

# # Usage example
# batch_size = 10
# seq_length = 40000  # Example sequence length
# input_tensor = torch.rand((batch_size, 1, seq_length))  # Example input (batch_size, channels, sequence_length)
# model = CombinedModel()
# output = model(input_tensor)
# print(output.shape)  # Should be [batch_size, output_size]
