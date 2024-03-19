import torch
import torch.nn as nn

class CombinedLSTMModel(nn.Module):
    def __init__(self, model1, model2, model3, model4, num_classes=5):
        super(CombinedLSTMModel, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4
        
        # Determine the size of the combined features
        self.feature_size = model1.MLP[-1].out_features + model2.MLP[-1].out_features + model3.MLP[-1].out_features + model4.MLP[-1].out_features
        # Add new layers to combine the features
        self.combined_layers = nn.Sequential(
            nn.Linear(self.feature_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        out1 = self.model1(x)
        out2 = self.model2(x)
        out3 = self.model3(x)
        out4 = self.model4(x)
        # out1 = self.model1.MLP(self.model1.lstm(x)[0]) 
        # out2 = self.model2.MLP(self.model2.lstm(x)[0])
        # out3 = self.model3.MLP(self.model3.lstm(x)[0])
        # out4 = self.model4.MLP(self.model4.lstm(x)[0])
        
        # Concatenate the features
        combined_features = torch.cat((out1, out2, out3, out4), dim=1)
        
        # Pass the combined features through the new layers
        output = self.combined_layers(combined_features)
        
        return output