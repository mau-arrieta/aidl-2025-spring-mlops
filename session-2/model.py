import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, activation_func='relu'):
        super(MyModel, self).__init__()
        
        # Select activation function
        if activation_func == 'relu':
            self.activation = nn.ReLU()
        elif activation_func == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation function {activation_func}")

        # Define layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 15)

    def forward(self, x):
        x = self.pool(self.activation(self.conv1(x)))  # Conv1 -> Activation -> MaxPool
        x = self.pool(self.activation(self.conv2(x)))  # Conv2 -> Activation -> MaxPool
        x = self.pool(self.activation(self.conv3(x)))  # Conv3 -> Activation -> MaxPool
        x = x.view(x.size(0), -1)  # Flatten
        x = self.activation(self.fc1(x))  # Fully Connected 1
        x = self.fc2(x)  # Output
        return x
