import torch
import torch.nn as nn
import torch.nn.functional as F

# Building Block
class BuildingBlock(nn.Module):
    def __init__(self, filters, s):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(filters[0])
        self.conv_x_1 = nn.Conv2d(in_channels=filters[0], out_channels=filters[1],
                                  kernel_size=(3, 3), stride=s, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(filters[1])
        self.conv_x_2 = nn.Conv2d(in_channels=filters[1], out_channels=filters[1],
                                  kernel_size=(3, 3), stride=1, padding=1, bias=False)

    def forward(self, x):
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv_x_1(x)
        x = self.bn2(x)
        x = F.relu(x)
        out = self.conv_x_2(x)
        return out


# BottleNeck Layers
class BottleNeck(nn.Module):
    def __init__(self, filters, s):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(filters[0])
        self.conv_x_1 = nn.Conv2d(in_channels=filters[0], out_channels=filters[1],
                                  kernel_size=(1, 1), stride=s, padding=0, bias=False)

        self.bn2 = nn.BatchNorm2d(filters[1])
        self.conv_x_2 = nn.Conv2d(in_channels=filters[1], out_channels=filters[1],
                                  kernel_size=(3, 3), stride=1, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(filters[1])
        self.conv_x_3 = nn.Conv2d(in_channels=filters[1], out_channels=filters[2],
                                  kernel_size=(1, 1), stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv_x_1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv_x_2(x)
        x = self.bn3(x)
        x = F.relu(x)
        out = self.conv_x_3(x)

        return out

# Identity Mapping
class IdentityMapping(nn.Module):
    def __init__(self, block, filters, stride):
        super().__init__()
        self.identity_filters = filters[-1:] + filters[1:]
        self.block = block(self.identity_filters, stride)

    def forward(self, x):
        x_id = x
        x = self.block(x)

        out = torch.add(x, x_id)
        return out

# Projection Mapping

class ProjectionMapping(nn.Module):
    def __init__(self, block, filters, stride):
        super().__init__()
        self.block = block(filters, stride)

        self.bn_proj = nn.BatchNorm2d(filters[0])
        self.projection = nn.Conv2d(in_channels=filters[0], out_channels=filters[-1],
                                    kernel_size=(1, 1), stride=stride, padding=0, bias=False)

    def forward(self, x):
        x_proj = x
        x = self.block(x)

        x_proj = self.bn_proj(x_proj)
        x_proj = F.relu(x_proj)
        x_proj = self.projection(x_proj)

        out = torch.add(x, x_proj)
        return out

# Complete ResNetV2
class ResNetV2(nn.Module):
    def __init__(self, block, filters,layers):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=64,
                                kernel_size=(7, 7), stride=2, padding=(3, 3))

        self.layer1 = self.get_layers(block, filters[0], 1, layers[0])
        self.layer2 = self.get_layers(block, filters[1], 2, layers[1])
        self.layer3 = self.get_layers(block, filters[2], 2, layers[2])
        self.layer4 = self.get_layers(block, filters[3], 2, layers[3])

        self.post_bn = nn.BatchNorm2d(filters[3][-1])

        self.avg_pool = nn.AvgPool2d(kernel_size=(2, 2))

        self.linear = nn.Linear(in_features=filters[3][-1], out_features=1000)

    def get_layers(self, block, filters, s, num_layers):
        layer = [ProjectionMapping(block, filters, s)]

        for i in range(num_layers - 1):
            layer.append(IdentityMapping(block, filters, 1))

        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.conv_1(x)
        x = F.max_pool2d(x, kernel_size=(3, 3), stride=2, padding=1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.post_bn(x)
        x = F.relu(x)

        x = F.adaptive_avg_pool2d(x, output_size=(1, 1))  # global average pooling
        x = x.view(-1)
        out = self.linear(x)
        return out

def resnet18V2():
    filters = [
        [64, 64],
        [64, 128],
        [128, 256],
        [256, 512]
    ]
    return ResNetV2(BuildingBlock, filters, [2, 2, 2, 2])

def resnet34V2():
    filters = [
        [64, 64],
        [64, 128],
        [128, 256],
        [256, 512]
    ]
    return ResNetV2(BuildingBlock, filters, [3, 4, 6, 3])

def resnet50V2():
    filters = [
        [64, 64, 256],
        [256, 128, 512],
        [512, 256, 1024],
        [1024, 512, 2048]
    ]
    return ResNetV2(BottleNeck, filters, [3, 4, 6, 3])

def resnet101V2():
    filters = [
        [64, 64, 256],
        [256, 128, 512],
        [512, 256, 1024],
        [1024, 512, 2048]
    ]
    return ResNetV2(BottleNeck, filters, [3, 4, 23, 3])

def resnet152V2():
    filters = [
        [64, 64, 256],
        [256, 128, 512],
        [512, 256, 1024],
        [1024, 512, 2048]
    ]
    return ResNetV2(BottleNeck, filters, [3, 8, 36, 3])
