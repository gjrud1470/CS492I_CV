import math
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from models import weights_init_kaiming, weights_init_classifier, ClassBlock


class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim, stride, dropRate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(input_dim, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_dim, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (input_dim == output_dim)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, input_dim, output_dim, block, stride, dropRate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, input_dim, output_dim, nb_layers, stride, dropRate, activate_before_residual)
    def _make_layer(self, block, input_dim, output_dim, nb_layers, stride, dropRate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and input_dim or output_dim, output_dim, i == 0 and stride or 1, dropRate, activate_before_residual))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=2, dropRate=0.0):
        super(WideResNet, self).__init__()
        channelNums = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, channelNums[0], kernel_size=3, stride=1, padding=1, bias=False)
        # blocks
        self.block1 = NetworkBlock(n, channelNums[0], channelNums[1], block, 1, dropRate, activate_before_residual=True)
        self.block2 = NetworkBlock(n, channelNums[1], channelNums[2], block, 2, dropRate)
        self.block3 = NetworkBlock(n, channelNums[2], channelNums[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(channelNums[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc = nn.Linear(channelNums[3], num_classes)
        self.fc.apply(weights_init_classifier)
        self.classifier = ClassBlock(channelNums[3], num_classes)
        self.classifier.apply(weights_init_classifier)
        self.channelNums = channelNums[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = nn.AdaptiveAvgPool2d((1,1))(out)
        fea =  out.view(out.size(0), -1)
        embed_fea = self.fc(fea)
        pred = self.classifier(fea)
        return embed_fea, pred