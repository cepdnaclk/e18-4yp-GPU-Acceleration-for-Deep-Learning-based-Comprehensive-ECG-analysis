from models.Inception1D import Inception1d
from torchinfo import summary  # pip install torchinfo

model = Inception1d(num_classes=1, input_channels=8, use_residual=True, ps_head=0.5, lin_ftrs_head=[128], kernel_size=40)


summary(model, input_size=(32, 8, 5000), depth=11)
