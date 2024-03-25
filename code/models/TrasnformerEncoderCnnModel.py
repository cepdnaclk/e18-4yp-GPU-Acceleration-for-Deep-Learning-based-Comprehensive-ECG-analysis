import torch
import torch.nn as nn
import math

# linear count is  19
# cnn count is  3
# trans count is  5
# posi count is  1

class TrasnformerEncoderCnnModel(nn.Module):
    def __init__(self, input_size, patch_size, num_layers, num_heads, dim_feedforward, output_size):
        super(TrasnformerEncoderCnnModel, self).__init__()
        self.embedding_layer = nn.Linear(patch_size, input_size)
        self.pos_encoder = PositionalEncoding(input_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads, dim_feedforward=dim_feedforward)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        # Simplified linear output layer
        self.linear_output = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_size)
        )
        
        # Deep CNN output layer
        self.cnn_output = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(16 * (input_size // 8), output_size)
        )
        
        self.patch_size = patch_size
        # self.reset_parameters()  # Initialize weights and biases
        # print("#=========Initialization to 1 and 0 done========")
 
    def reset_parameters(self):
    # Initialize weights to 1 and biases to 0
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.TransformerEncoderLayer):
                # Initialize the layers in the TransformerEncoderLayer
                nn.init.constant_(m.linear1.weight, 1)
                nn.init.constant_(m.linear1.bias, 0)
                nn.init.constant_(m.linear2.weight, 1)
                nn.init.constant_(m.linear2.bias, 0)
                nn.init.constant_(m.self_attn.in_proj_weight, 1)
                nn.init.constant_(m.self_attn.in_proj_bias, 0)
                nn.init.constant_(m.self_attn.out_proj.weight, 1)
                nn.init.constant_(m.self_attn.out_proj.bias, 0)
            elif isinstance(m, PositionalEncoding):
                # No need to initialize the PositionalEncoding module
                # since it doesn't have any learnable parameters.
                pass

    def forward(self, x):
        batch_size, _ = x.size()
        num_patches = x.size(1) // self.patch_size
        x = x.view(batch_size, num_patches, self.patch_size)
        x = self.embedding_layer(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        
        # Apply linear output layer
        linear_out = self.linear_output(x)
        
        # Apply CNN output layer
        cnn_out = self.cnn_output(x.unsqueeze(1))
        
        # Combine the outputs (e.g., addition or concatenation)
        output = linear_out + cnn_out
        
        return output

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