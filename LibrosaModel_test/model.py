import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEmotionClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CNNEmotionClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.fc1 = nn.Linear(32 * 9 * 23, 128)  # Conv 후 크기 (입력이 40x100 기준)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (B, 16, 38, 98) → (B, 16, 19, 49)
        x = self.pool(F.relu(self.conv2(x)))  # (B, 32, 9, 23)
        x = x.view(-1, 32 * 9 * 23)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x