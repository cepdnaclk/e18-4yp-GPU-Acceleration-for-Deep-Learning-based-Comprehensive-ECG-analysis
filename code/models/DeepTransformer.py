import torch
import torch.nn as nn
from fastai.layers import *
from fastai.core import *

##############################################################################################################################################
# utility functions

class AdaptiveConcatPool1d(nn.Module):
    "Layer that concats `AdaptiveAvgPool1d` and `AdaptiveMaxPool1d`."
    def __init__(self, sz: Optional[int] = None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        sz = sz or 1
        self.ap, self.mp = nn.AdaptiveAvgPool1d(sz), nn.AdaptiveMaxPool1d(sz)
        
    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)

def create_head1d(nf: int, nc: int, lin_ftrs: Optional[Collection[int]] = None, ps: Floats = 0.5, bn_final: bool = False, bn: bool = True, act: str = "relu", concat_pooling: bool = True):
    "Model head that takes `nf` features, runs through `lin_ftrs`, and about `nc` classes; added bn and act here"
    lin_ftrs = [2 * nf if concat_pooling else nf, nc] if lin_ftrs is None else [2 * nf if concat_pooling else nf] + lin_ftrs + [nc]
    ps = listify(ps)
    if len(ps) == 1: ps = [ps[0] / 2] * (len(lin_ftrs) - 2) + ps
    actns = [nn.ReLU(inplace=True) if act == "relu" else nn.ELU(inplace=True)] * (len(lin_ftrs) - 2) + [None]
    layers = [AdaptiveConcatPool1d() if concat_pooling else nn.MaxPool1d(2), Flatten()]
    for ni, no, p, actn in zip(lin_ftrs[:-1], lin_ftrs[1:], ps, actns):
        layers += bn_drop_lin(ni, no, bn, p, actn)
    if bn_final: layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    return nn.Sequential(*layers)

##############################################################################################################################################

class TransformerBlock1d(nn.Module):
    def __init__(self, ni: int, nh: int, nheads: int, nlayers: int, dropout: float = 0.5):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=nh, nhead=nheads, dim_feedforward=nh, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        
    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, channels, seq_len) to (batch, seq_len, channels)
        x = self.transformer(x)
        return x.transpose(1, 2)  # back to (batch, channels, seq_len)

class TransformerBackbone(nn.Module):
    def __init__(self, input_channels: int, nh: int, nheads: int, nlayers: int, depth: int, dropout: float = 0.5):
        super().__init__()
        self.input_proj = nn.Linear(input_channels, nh)
        self.depth = depth
        self.transformer_blocks = nn.ModuleList([TransformerBlock1d(nh, nh, nheads, nlayers, dropout) for _ in range(depth)])
        
    def forward(self, x):
        x = self.input_proj(x.transpose(1, 2)).transpose(1, 2)  # Project input to nh dimensions
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        return x

class TransformerModel1d(nn.Module):
    def __init__(self, num_classes: int = 5, input_channels: int = 8, nh: int = 128, nheads: int = 8, nlayers: int = 2, depth: int = 6, dropout: float = 0.5, lin_ftrs_head: Optional[Collection[int]] = None, ps_head: Floats = 0.5, bn_final_head: bool = False, bn_head: bool = True, act_head: str = "relu", concat_pooling: bool = True):
        super().__init__()
        layers = [TransformerBackbone(input_channels=input_channels, nh=nh, nheads=nheads, nlayers=nlayers, depth=depth, dropout=dropout)]
        
        # Head
        head = create_head1d(nh, nc=num_classes, lin_ftrs=lin_ftrs_head, ps=ps_head, bn_final=bn_final_head, bn=bn_head, act=act_head, concat_pooling=concat_pooling)
        layers.append(head)
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# Example usage:
model = TransformerModel1d(num_classes=5, input_channels=8, nh=128, nheads=8, nlayers=2, depth=6)
x = torch.randn(16, 8, 100)  # Batch size: 16, Channels: 8, Sequence length: 100
output = model(x)
print(output.shape)  # Should be (16, 5)
