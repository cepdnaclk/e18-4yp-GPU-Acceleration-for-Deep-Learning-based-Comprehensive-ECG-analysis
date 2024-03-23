import torch.nn as nn
from torchaudio.models import wav2vec2_base


class Wave2Vec2(nn.Module):
    def __init__(self):
        super(Wave2Vec2, self).__init__()

        # Initialize the Wav2Vec2Model
        self.wav2vec2 = wav2vec2_base()

        # Freeze the Wav2Vec2Model weights
        # for param in self.wav2vec2.parameters():
        #     param.requires_grad = False

        # Define the remaining layers
        self.all = nn.Sequential(
            nn.Linear(768, 10),  # Adjust the input size based on the output size of Wav2Vec2Model
            nn.ReLU(),
            nn.Linear(10, 1),
        )

    def forward(self, x):
        # Pass the input through the Wav2Vec2Model
        x = self.wav2vec2.extract_features(x)[0][-1]  # Get the last layer output
        x = x[:, -1, :]

        # Pass the Wav2Vec2Model output through the remaining layers
        x = self.all(x)
        return x
