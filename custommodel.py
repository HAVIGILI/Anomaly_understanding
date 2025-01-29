import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=3, stride=1, padding=1)    # [32, 32]
        self.bn1 = nn.BatchNorm2d(9)
        self.conv2 = nn.Conv2d(in_channels=9, out_channels=15, kernel_size=5, stride=1, padding=2)  # [32, 32]
        self.bn2 = nn.BatchNorm2d(15)
        self.conv3 = nn.Conv2d(in_channels=15, out_channels=20, kernel_size=7, stride=7, padding=3) # [5, 5]
        self.bn3 = nn.BatchNorm2d(20)
        self.conv4 = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=1, stride=1, padding=0) # [5, 5]
        self.bn4 = nn.BatchNorm2d(10)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)  # [3, 3]
        # Fully connected layer
        self.fc = nn.Linear(10 * 3 * 3, output_size)  # 90 features

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # [9, 32, 32]
        x = F.relu(self.bn2(self.conv2(x)))  # [15, 32, 32]
        x = F.relu(self.bn3(self.conv3(x)))  # [20, 5, 5]
        x = F.relu(self.bn4(self.conv4(x)))  # [10, 5, 5]
        x = self.maxpool1(x)                  # [10, 3, 3]
        x = torch.flatten(x, start_dim=1)     # [batch_size, 90]
        x = self.fc(x)                         # [batch_size, output_size]
        return x

# Example Usage
if __name__ == "__main__":
    model = CustomModel(input_size=(3, 32, 32), output_size=10)
    print(model)
    # Create a dummy input tensor
    input_tensor = torch.randn(8, 3, 32, 32)  # [batch_size, channels, height, width]
    output = model(input_tensor)
    print("Output shape:", output.shape)  # Expected: [8, 10]
    print(output)
    probs = F.softmax(output, dim = 1)
    print(probs)
    hej, predictions = torch.max(probs, dim = 1)
    print(predictions)
    print(hej)