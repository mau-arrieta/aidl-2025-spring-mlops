
import torch.nn as nn

class MyModel(nn.Module):

    def __init__(self):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # MaxPooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # Adjust based on image size after conv layers
        self.fc2 = nn.Linear(128, 15)

    def forward(self, x):
        x = self.pool(nn.ReLU(self.conv1(x)))  # Conv1 -> ReLU -> MaxPool
        x = self.pool(nn.ReLU(self.conv2(x)))  # Conv2 -> ReLU -> MaxPool
        x = self.pool(nn.ReLU(self.conv3(x)))  # Conv3 -> ReLU -> MaxPool
        
        # Flatten before fully connected layers
        x = x.view(x.size(0), -1)  # Flatten tensor

        x = nn.ReLU(self.fc1(x))  # Fully Connected 1
        x = self.fc2(x)  # Output layer (logits)

        return x