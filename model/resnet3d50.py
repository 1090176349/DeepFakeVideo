# 文件: model/resnet3d50.py

import torch
import torch.nn as nn
from config import config

def conv3x3x3(in_planes, out_planes, stride=1):
    """3x3x3 卷积，保持尺寸"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class Bottleneck3D(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        """
        Bottleneck 模块，采用 1x1x1、3x3x3、1x1x1 的结构，
        expansion 表示输出通道相对于 planes 的放大倍数
        """
        super(Bottleneck3D, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out

class ResNet3D50(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        """
        构造 ResNet3D-50 模型。
        参数:
          - block: 基本构建模块，这里为 Bottleneck3D
          - layers: 每个层包含的 block 数量，典型的配置为 [3,4,6,3]
          - num_classes: 最终全连接层输出的类别数
        """
        super(ResNet3D50, self).__init__()
        self.inplanes = 64
        # Stem 部分，7x7x7 卷积
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1,2,2), padding=(3,3,3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # 四个层次
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 输入 x: [B, 3, T, H, W]
        x = self.conv1(x)   # [B, 64, T, H/2, W/2]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # [B, 64, T, H/4, W/4]
        x = self.layer1(x)  # [B, 256, T, H/4, W/4]
        x = self.layer2(x)  # [B, 512, T, H/8, W/8]
        x = self.layer3(x)  # [B, 1024, T, H/16, W/16]
        x = self.layer4(x)  # [B, 2048, T, H/32, W/32]
        x = self.avgpool(x) # [B, 2048, 1, 1, 1]
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def resnet3d50(**kwargs):
    """
    工厂函数，构造 ResNet3D-50 模型
    默认层数配置为 [3, 4, 6, 3]
    """
    model = ResNet3D50(Bottleneck3D, [3, 4, 6, 3], **kwargs)
    return model


if __name__ == '__main__': 

    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    model = resnet3d50(num_classes=config.NUM_CLASSES)
    x = torch.randn(16, 3, 32, 112, 112).to(device)
    logits = model(x)
    print("Logits shape:", logits.shape)

