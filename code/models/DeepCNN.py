# import torch.nn as nn
# import torch.nn.functional as F


# # class DeepCNN(nn.Module):
# #     def __init__(self):
# #         super(DeepCNN, self).__init__()

# #         # Convolutional layers
# #         self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
# #         self.conv2 = nn.Conv1d(16, 1, kernel_size=3, stride=1, padding=1)

# #         # Fully connected layers
# #         self.fc1 = nn.Linear(40000, 128)  
# #         self.fc2 = nn.Linear(128, 1)  

# #     def forward(self, x):
# #         # Apply convolutional layers with activation functions
# #         x = F.relu(self.conv1(x))
# #         x = F.relu(self.conv2(x))

# #         # Flatten the tensor for fully connected layers
# #         x = x.view(x.size(0), -1)

# #         # Apply fully connected layers with activation functions
# #         x = F.relu(self.fc1(x))
# #         x = self.fc2(x)

# #         return x

# # # import torch.nn as nn
# # # import torch.nn.functional as F

# # # class DeepCNN(nn.Module):
# # #     # def __init__(self):
# # #     #     super(DeepCNN, self).__init__()
# # #     #     # Convolutional layers
# # #     #     self.conv1 = nn.Conv1d(1, 40000, kernel_size=5, stride=1, padding=2)
# # #     #     self.conv2 = nn.Conv1d(40000, 20000, kernel_size=5, stride=1, padding=2)
# # #     #     self.conv3 = nn.Conv1d(20000, 10000, kernel_size=5, stride=1, padding=2)
# # #     #     self.conv4 = nn.Conv1d(10000, 1000, kernel_size=5, stride=1, padding=2)
# # #     #     self.conv5 = nn.Conv1d(256, 512, kernel_size=5, stride=1, padding=2)
# # #     #     self.conv6 = nn.Conv1d(512, 1, kernel_size=5, stride=1, padding=2)

# # #     #     # Batch normalization layers
# # #     #     self.bn1 = nn.BatchNorm1d(32)
# # #     #     self.bn2 = nn.BatchNorm1d(64)
# # #     #     self.bn3 = nn.BatchNorm1d(128)
# # #     #     self.bn4 = nn.BatchNorm1d(256)
# # #     #     self.bn5 = nn.BatchNorm1d(512)
# # #     #     self.bn6 = nn.BatchNorm1d(1)  # Added this line

# # #     #     # Dropout layers
# # #     #     self.dropout1 = nn.Dropout(0.2)
# # #     #     self.dropout2 = nn.Dropout(0.3)
# # #     #     self.dropout3 = nn.Dropout(0.4)
# # #     #     self.dropout4 = nn.Dropout(0.5)
# # #     #     self.dropout5 = nn.Dropout(0.6)

# # #     # def forward(self, x):
# # #     #     # Apply convolutional layers with batch normalization, activation functions, and dropout
# # #     #     x = self.dropout1(F.relu(self.bn1(self.conv1(x))))
# # #     #     x = self.dropout2(F.relu(self.bn2(self.conv2(x))))
# # #     #     x = self.dropout3(F.relu(self.bn3(self.conv3(x))))
# # #     #     x = self.dropout4(F.relu(self.bn4(self.conv4(x))))
# # #     #     x = self.dropout5(F.relu(self.bn5(self.conv5(x))))
# # #     #     x = self.bn6(self.conv6(x))  # Batch normalization for the final output

# # #     #     return x
# # #     def __init__(self):
# # #         super(DeepCNN, self).__init__()

# # #         # Convolutional layers
# # #         self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
# # #         self.conv2 = nn.Conv1d(16, 1, kernel_size=3, stride=1, padding=1)

# # #         # Fully connected layers
# # #         self.fc1 = nn.Linear(40000, 128)  
# # #         self.fc2 = nn.Linear(128, 1)  

# # #     def forward(self, x):
# # #         # Apply convolutional layers with activation functions
# # #         x = F.relu(self.conv1(x))
# # #         x = F.relu(self.conv2(x))

# # #         # Flatten the tensor for fully connected layers
# # #         x = x.view(x.size(0), -1)

# # #         # Apply fully connected layers with activation functions
# # #         x = F.relu(self.fc1(x))
# # #         x = self.fc2(x)

# # #         return x


# # # import torch
# # # import torch.nn as nn
# # # import torch.nn.functional as F

# # # class DeepCNN(nn.Module):
# # #     def __init__(self):
# # #         super(DeepCNN, self).__init__()
# # #         # Convolutional layers
# # #         self.conv1 = nn.Conv1d(1, 40000, kernel_size=5, stride=1, padding=2)
# # #         self.bn1 = nn.BatchNorm1d(40000)
# # #         self.conv2 = nn.Conv1d(40000, 64, kernel_size=5, stride=1, padding=2)
# # #         self.bn2 = nn.BatchNorm1d(20000)
# # #         self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
# # #         self.bn3 = nn.BatchNorm1d(10000)
# # #         self.conv4 = nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2)
# # #         self.bn4 = nn.BatchNorm1d(5000)
# # #         self.conv5 = nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=2)
# # #         self.bn5 = nn.BatchNorm1d(2500)
# # #         self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

# # #         # Fully connected layers
# # #         self.fc1 = nn.Linear(1250, 512)  # Adjust the linear layer input size accordingly
# # #         self.fc2 = nn.Linear(512, 128)
# # #         self.fc3 = nn.Linear(128, 1)

# # #     def forward(self, x):
# # #         # Apply convolutional layers with activation functions and pooling
# # #         x = F.relu(self.bn1(self.conv1(x)))
# # #         x = self.pool(x)
# # #         x = F.relu(self.bn2(self.conv2(x)))
# # #         x = self.pool(x)
# # #         x = F.relu(self.bn3(self.conv3(x)))
# # #         x = self.pool(x)
# # #         x = F.relu(self.bn4(self.conv4(x)))
# # #         x = self.pool(x)
# # #         x = F.relu(self.bn5(self.conv5(x)))
# # #         x = self.pool(x)

# # #         # Flatten the tensor for fully connected layers
# # #         x = x.view(x.size(0), -1)

# # #         # Apply fully connected layers with activation functions
# # #         x = F.relu(self.fc1(x))
# # #         x = F.relu(self.fc2(x))
# # #         x = self.fc3(x)
# # #         return x






# # ------------------------- WORKING ----------------------------------------

# import torch.nn as nn
# import torch.nn.functional as F

# class DeepCNN(nn.Module):
#     def __init__(self):
#         super(DeepCNN, self).__init__()

#         # Assuming the input sequence length is N (e.g., 40000 as hinted by your original fc1 layer)
#         N = 39998#40000
        
#         # Convolutional layers
#         self.conv1 = nn.Conv1d(1, 16, kernel_size=500, stride=1, padding=249)
#         self.conv2 = nn.Conv1d(16, 1, kernel_size=500, stride=1, padding=249)

#         # Calculate the output size after convolutions
#         # Output size formula for padding 'same' and stride 1: out_length = in_length
#         # Hence, after both convolutions, output size remains N
#         conv_output_size = N  # Adjust based on your specific inputs and architecture

#         # Fully connected layers
#         self.fc1 = nn.Linear(conv_output_size, 128)  # Adjust input size according to the conv output size
#         self.fc2 = nn.Linear(128, 1)

#     def forward(self, x):
#         # Apply convolutional layers with activation functions
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))

#         # Flatten the tensor for fully connected layers
#         x = x.view(x.size(0), -1)

#         # Apply fully connected layers with activation functions
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)

#         return x



import torch.nn as nn
import torch.nn.functional as F

class DeepCNN(nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()

        # Initial input size is presumed to be 40000 (or as per your data context)
        # Convolutional layers with specified kernel sizes and adjusted stride/padding
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5000, stride=4, padding=2498) # output size approx 10000
        self.conv2 = nn.Conv1d(64, 32, kernel_size=2500, stride=4, padding=1249) # output size approx 2500
        self.conv3 = nn.Conv1d(32, 16, kernel_size=1000, stride=5, padding=499) # output size approx 500
        self.conv4 = nn.Conv1d(16, 8, kernel_size=500, stride=5, padding=249) # output size approx 100
        self.conv5 = nn.Conv1d(8, 4, kernel_size=250, stride=2, padding=124) # output size approx 50
        self.conv6 = nn.Conv1d(4, 2, kernel_size=100, stride=2, padding=49) # output size approx 25
        self.conv7 = nn.Conv1d(2, 1, kernel_size=50, stride=1, padding=24) # output size approx 25
        self.conv8 = nn.Conv1d(1, 1, kernel_size=25, stride=1, padding=12) # output size approx 25

        # Fully connected layers
        self.fc1 = nn.Linear(24, 128)  # Adjusted the input size based on the output size of the last convolutional layer
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # Apply convolutional layers with activation functions
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))

        # Flatten the tensor for fully connected layers
        x = x.view(x.size(0), -1)

        # Apply fully connected layers with activation functions
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
