import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Se(nn.Module):
    def __init__(self, in_channel, reduction=16):
        super(Se, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Sequential(
            nn.Linear(in_features=in_channel, out_features=in_channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=in_channel // reduction, out_features=in_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.pool(x)
        out = self.fc(out.view(out.size(0), -1))
        out = out.view(x.size(0), x.size(1), 1, 1)
        return out * x


class SEWeightModule(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight


class ECA(nn.Module):
    def __init__(self, in_channel, gamma=2, b=1):
        super(ECA, self).__init__()
        k = int(abs(math.log(in_channel, 2 + b) / gamma))
        kernel_size = k if k % 2 else k + 1
        padding = kernel_size // 2
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.pool(x)
        out = out.view(x.size(0), 1, x.size(1))
        out = self.conv(out)
        out = out.view(x.size(0), x.size(1), 1, 1)
        return out * x


class CALayer(nn.Module):  # Channel Attention (CA) Layer
    def __init__(self, in_channels, reduction=16, pool_types=['avg', 'max']):
        super().__init__()
        self.pool_list = ['avg', 'max']
        self.pool_types = pool_types
        self.in_channels = in_channels
        self.Pool = [nn.AdaptiveAvgPool2d(
            1), nn.AdaptiveMaxPool2d(1, return_indices=False)]
        self.conv_ca = nn.Sequential(
            nn.Conv2d(in_channels, in_channels //
                      reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction,
                      in_channels, 1, padding=0, bias=True)
        )

    def forward(self, x):
        for (i, pool_type) in enumerate(self.pool_types):
            pool = self.Pool[self.pool_list.index(pool_type)](x)
            channel_att_raw = self.conv_ca(pool)
            if i == 0:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum += channel_att_raw
        scale = F.sigmoid(channel_att_sum)
        return x * scale


class SALayer(nn.Module):  # Spatial Attention Layer
    def __init__(self):
        super().__init__()
        self.conv_sa = nn.Sequential(
            nn.Conv2d(2, 1, 3, 1, 1, bias=False),
            nn.BatchNorm2d(1, momentum=0.01),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_compress = torch.cat(
            (torch.max(x, 1, keepdim=True)[0], torch.mean(x, dim=1, keepdim=True)), dim=1)
        scale = self.conv_sa(x_compress)
        return x * scale


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=2, pool_types=['avg', 'max']):
        super().__init__()
        self.CALayer = CALayer(
            in_channels, reduction, pool_types)
        self.SALayer = SALayer()

    def forward(self, x):
        x_out = self.CALayer(x)
        x_out = self.SALayer(x_out)
        return x_out
