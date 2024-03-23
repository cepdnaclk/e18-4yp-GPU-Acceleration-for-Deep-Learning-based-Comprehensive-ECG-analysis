import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from models.DeepViT import DeepViT

class CnnDeepViT(nn.Module):
    def __init__(self, image_size=256, num_classes=1):
        super(CnnDeepViT, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(256 * (image_size // 16) * (image_size // 16), 1024)
        self.deepvit = DeepViT(
            image_size=image_size,
            patch_size=32,
            num_classes=num_classes,
            dim=1024,
            depth=6,
            heads=9,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )

    def forward(self, x):
        x = x.squeeze(0)  # Remove the extra dimension
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = self.maxpool(x)
        x = F.relu(self.conv4(x))
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = x.unsqueeze(0)  # Add batch dimension
        x = self.deepvit(x)
        return x


