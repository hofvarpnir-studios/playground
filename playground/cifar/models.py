import torch
import torch.nn as nn
import torch.nn.functional as F
import gin


@gin.configurable
class MLPNet(nn.Module):
    def __init__(self, hidden_dim: int = 4096):
        super().__init__()
        self.linear1 = nn.Linear(3 * 32 * 32, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, int(hidden_dim / 2))
        self.linear3 = nn.Linear(int(hidden_dim / 2), 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.flatten(x, 1)  # flatten all dimensions except batch
        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        return out


@gin.configurable
class ConvNet(nn.Module):
    def __init__(self, filters: int = 16, kernel: int = 3, hidden_dim: int = 512):
        super().__init__()
        self.conv1 = nn.Conv2d(3, filters, kernel, 1, padding=1)
        self.conv2 = nn.Conv2d(filters, filters * 2, kernel, 1, padding=1)
        self.conv3 = nn.Conv2d(filters * 2, filters * 4, kernel, 1, padding=1)
        self.fc1 = nn.Linear(4 * 4 * filters * 4, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
