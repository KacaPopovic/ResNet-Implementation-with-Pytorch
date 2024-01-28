import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.max_pool = nn.MaxPool2d(3, 2)
        self.res1 = ResBlock(64, 64, 1)
        self.res2 = ResBlock(64, 128, 2)
        self.res3 = ResBlock(128, 256, 2)
        self.res4 = ResBlock(256, 512, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        flag = False
        out = self.conv1(x)
        if out.dim()==3:
            out = out.unsqueeze(0)
            flag = True
        out = self.bn1(out)
        out = F.relu(out)
        out = self.max_pool(out)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.global_pool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)
        out = torch.sigmoid(out)
        if flag:
            out = out.squeeze(0)
        return out
