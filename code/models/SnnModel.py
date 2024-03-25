import torch
import torch.nn as nn
import snntorch as spnn

class SnnModel(nn.Module):
    def __init__(self):
        super(SnnModel, self).__init__()
        
        # Define the input and hidden layers
        self.input_layer = spnn(40000)
        self.hidden_layer1 = spnn.SpikingRNN(40000, 128, num_steps=100)
        self.hidden_layer2 = spnn.SpikingRNN(128, 64, num_steps=100)
        
        # Define the output layer
        self.output_layer = spnn.Sigmoid(64, 1)
        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.output_layer(x)
        return x