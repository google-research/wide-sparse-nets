'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['resnet', 'resnet18', 'resnet18r', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class resnet(nn.Module):
    def __init__(self, block, num_blocks, num_out_conv1, out_ratio, num_classes=10, num_in_channels=3):
        super(resnet, self).__init__()
        self.in_planes = num_out_conv1

        self.conv1 = nn.Conv2d(num_in_channels, num_out_conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_out_conv1)
        self.layer1 = self._make_layer(block, out_ratio[0]*num_out_conv1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, out_ratio[1]*num_out_conv1, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, out_ratio[2]*num_out_conv1, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, out_ratio[3]*num_out_conv1, num_blocks[3], stride=2)
        self.linear = nn.Linear(out_ratio[3]*num_out_conv1*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet18(num_in_channels, num_classes, num_out_conv1=64, out_ratio=[1,2,4,8]):
    return resnet(BasicBlock, num_blocks=[2,2,2,2], num_out_conv1=num_out_conv1, out_ratio=out_ratio, 
                              num_classes=num_classes, num_in_channels=num_in_channels)

def resnet18r(num_in_channels, num_classes, num_out_conv1=64, out_ratio=[1,2,4,8]):
    return resnet(BasicBlock, num_blocks=[1,1,1,1], num_out_conv1=num_out_conv1, out_ratio=out_ratio, num_classes=num_classes, num_in_channels=num_in_channels)

def resnet34():
    return resnet(BasicBlock, [3,4,6,3])

def resnet50():
    return resnet(Bottleneck, [3,4,6,3])

def resnet101():
    return resnet(Bottleneck, [3,4,23,3])

def resnet152():
    return resnet(Bottleneck, [3,8,36,3])


def test():
    net = resnet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
