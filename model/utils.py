import torch
import numpy as np
# from tensorboardX import SummaryWriter
from config import opt
from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, ReLU, Upsample, MaxPool2d, AvgPool2d
from torch.nn import functional as F
# from vis.utils import get_points_from_heatmaps


def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=False):
    """3x3的卷积层"""
    return Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                  stride=stride, padding=padding, bias=bias)


class HGBlock_expand(Module):
    """
    扩展版的HGBlock, 在他的后面又加上了一些处理
    """
    def __init__(self, feature_channels=256, num_classes=68, depth=4):
        """
        :param feature_channels: (int) 网络中最大的feature channels
        :param num_classes: (int) 想要预测的关键点的数量
        :param depth: (int) HG网络的单向的深度
        """
        super(HGBlock_expand, self).__init__()
        self.hg = HGBlock(feature_channels=feature_channels, depth=depth)
        self.after_hg = Sequential(ResBlock(feature_channels, feature_channels),
                                   Conv2d(feature_channels, feature_channels, 1, 1, padding=0),
                                   BatchNorm2d(num_features=feature_channels),
                                   ReLU(True))
        self.left_conv = Conv2d(feature_channels, feature_channels, 1, 1)
        self.class_conv = Conv2d(feature_channels, num_classes, 1, 1)
        self.conv_after_out = Conv2d(num_classes, feature_channels, 1, 1)

    def forward(self, x):

        residual = x
        x = self.hg(x)
        x = self.after_hg(x)
        xl = self.left_conv(x)
        class_tensor = self.class_conv(x)
        xr = self.conv_after_out(class_tensor)
        output = residual + xl + xr

        return class_tensor, output


class HGBlock(Module): # 根据ELG的论文中的结构来的
    def __init__(self, feature_channels=256, depth=4):
        """
        :param feature_channels: 在HGNet里面传播的feature map的通道数
        :param depth:  每一个HGNet的一块的downsample的次数
        """
        super(HGBlock, self).__init__()
        self.feature_channels = feature_channels
        self.layer_num = depth
        # down sample part
        for i in range(depth):
            setattr(self, 'down_res_{}'.format(i), ResBlock(feature_channels, feature_channels))
            # setattr(self, 'down_pool_{}'.format(i), MaxPool2d(kernel_size=(2, 2)))
        # parallel part
        for i in range(1):
            setattr(self, 'paral_res_{}'.format(i), ResBlock(feature_channels, feature_channels))
        # up sample part
        for i in range(depth):
            setattr(self, 'up_res_{}'.format(i), ResBlock(feature_channels, feature_channels))
            # setattr(self, 'up_sample_{}'.format(i), Upsample(scale_factor=2))
        # short cut part
        for i in range(depth):
            setattr(self, 'shortcut_res_{}'.format(i), ResBlock(feature_channels, feature_channels))
        # 在forward 函数里面实现

    def forward(self, x):
        cache_stack = []  # 用来保存前面的值，short cut
        # down sample part
        for i in range(self.layer_num):
            cache_stack.append(x)
            x = F.avg_pool2d(x, 2, stride=2)
            x = getattr(self, 'down_res_{}'.format(i))(x)

        # parallel part
        for i in range(1):
            x = getattr(self, 'paral_res_{}'.format(i))(x)
        # up sample part
        for i in range(self.layer_num):
            residual = cache_stack.pop(-1)
            residual = getattr(self, 'shortcut_res_{}'.format(i))(residual)
            x = getattr(self, 'up_res_{}'.format(i))(x)
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = x + residual
        return x


class ResBlock(Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        assert out_channels % 4 == 0, '输出的channels必须要能够被4整除'
        self.bn1 = BatchNorm2d(num_features=in_channels)
        self.conv1 = conv3x3(in_channels=in_channels, out_channels=int(out_channels / 2))
        self.bn2 = BatchNorm2d(num_features=int(out_channels / 2))
        self.conv2 = conv3x3(in_channels=int(out_channels / 2), out_channels=int(out_channels / 4))
        self.bn3 = BatchNorm2d(num_features=int(out_channels/4))
        self.conv3 = conv3x3(in_channels=int(out_channels / 4), out_channels=int(out_channels / 4))

        if in_channels != out_channels:
            self.conv1x1 = Sequential(
                BatchNorm2d(num_features=in_channels),
                ReLU(inplace=True),
                Conv2d(in_channels=in_channels, out_channels=out_channels,
                       kernel_size=1, stride=1, bias=False)
            )
        else:
            self.conv1x1 = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, inplace=True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, inplace=True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.conv1x1 is not None:
            residual = self.conv1x1(residual)

        out3 = out3 + residual

        return out3


class DownSampleConvBlock(Module):
    def __init__(self, in_channels, out_channels):
        super(DownSampleConvBlock, self).__init__()

        self.block = Sequential(
            Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1),
            BatchNorm2d(num_features=out_channels),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2))
        )

    def forward(self, x):
        out = self.block.forward(x)
        return out


class GazeEstimatBlock(Module):
    """
    直接根据HGnet 生成的feature map 来预测look vector
    """
    def __init__(self, feature_channels=128, num_classes=2):
        super(GazeEstimatBlock, self).__init__()
        self.feature_channels = feature_channels

        self.downsample_block = Sequential(
            ResBlock(feature_channels, int(feature_channels/2)),
            MaxPool2d(kernel_size=(2, 2)),
            ResBlock(int(feature_channels/2), int(feature_channels/4)),
            MaxPool2d(kernel_size=(2, 2)),
            ResBlock(int(feature_channels/4), int(feature_channels/8)),
            MaxPool2d(kernel_size=(2, 2)),
            Conv2d(int(feature_channels/8), out_channels=num_classes, kernel_size=1, stride=1)
        )
        # 最后输出的每一个channel 代表一个向量的维度

        self.glob_average_pool = AvgPool2d(kernel_size=(12, 20))  # 2 × 1 × 1

    def forward(self, x):

        # assert x.size(2) == 96 and x.size(3) == 160, '输入的feature的HxW必须是96x160'
        x = self.downsample_block.forward(x)
        out = self.glob_average_pool.forward(x)

        return out
