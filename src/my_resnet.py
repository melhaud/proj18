'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

    name    | # layers | # params
ResNet-20   |    20    |   0.27M
ResNet-32   |    32    |   0.46M
ResNet-44   |    44    |   0.66M
ResNet-56   |    56    |   0.85M
ResNet-110  |    110   |   1.7M
ResNet-1202 |    1202  |   19.4M

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Nikita Balabin.
'''
import torch
import torch.nn as nn

from torch.functional import F
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import BasicBlock

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

model_urls = {
    'resnet20': None,
    'resnet32': None,
    'resnet44': None,
    'resnet56': None,
    'resnet110': None,
    'resnet1202': None,
}

class Padding(nn.Module):
    def __init__(self, inplanes, planes, stride):
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = (planes - inplanes) // 2
        
    def forward(self, x):
        return F.pad(x[:, :, ::self.stride[0], ::self.stride[1]],
                     (0, 0, 0, 0, self.padding, self.padding), "constant", 0.)
    
    def extra_repr(self):
        s = ('{inplanes}, {planes}, stride={stride}')
        return s.format(**self.__dict__)

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Padding(self.inplanes, planes, stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def resnet20(pretrained=False, progress=True, **kwargs):
    r"""ResNet-20 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet20', BasicBlock, [3, 3, 3], pretrained, progress,
                   **kwargs)

def resnet32(pretrained=False, progress=True, **kwargs):
    r"""ResNet-32 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet32', BasicBlock, [5, 5, 5], pretrained, progress,
                   **kwargs)

def resnet44(pretrained=False, progress=True, **kwargs):
    r"""ResNet-44 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet44', BasicBlock, [7, 7, 7], pretrained, progress,
                   **kwargs)

def resnet56(pretrained=False, progress=True, **kwargs):
    r"""ResNet-56 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet56', BasicBlock, [9, 9, 9], pretrained, progress,
                   **kwargs)

def resnet110(pretrained=False, progress=True, **kwargs):
    r"""ResNet-110 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet110', BasicBlock, [18, 18, 18], pretrained, progress,
                   **kwargs)

def resnet1202(pretrained=False, progress=True, **kwargs):
    r"""ResNet-1202 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet1202', BasicBlock, [200, 200, 200], pretrained, progress,
                   **kwargs)
