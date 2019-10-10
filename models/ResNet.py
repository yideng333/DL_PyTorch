import torch
from torch import nn
import torch.nn.functional as F


# 残差块
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )

        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, x):
        Y = self.conv(x)
        if self.conv3:
            x = self.conv3(x)
        return F.relu(Y + x)


class ResNet_18(nn.Module):
    def __init__(self, args):
        super(ResNet_18, self).__init__()
        b1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # 4个残差模块，每个模块使用2个残差块
        b2 = nn.Sequential(
            self.resnet_block(64, 64, 2, first_block=True),
            self.resnet_block(64, 128, 2),
            self.resnet_block(128, 256, 2),
            self.resnet_block(256, 512, 2),
            nn.AvgPool2d(kernel_size=7))

        self.conv = nn.Sequential(b1, b2)
        self.fc = nn.Linear(512, 10)

    def resnet_block(self, in_channels, out_channels, num_residuals, first_block=False):
        if first_block:
            assert in_channels == out_channels  # 第一个模块的通道数同输入通道数一致
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
            else:
                blk.append(Residual(out_channels, out_channels))
        return nn.Sequential(*blk)

    def forward(self, x):
        x = self.conv(x)
        output = self.fc(x.view(x.shape[0], -1))
        return output
