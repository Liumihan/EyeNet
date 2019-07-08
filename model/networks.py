import torch
import numpy as np
from torch import nn
from torch.nn import Module, Sequential
from torch.nn import Conv2d, BatchNorm2d, ReLU, Upsample, MaxPool2d, AvgPool2d

def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=False):
    """3x3的卷积层"""
    return Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                  stride=stride, padding=padding, bias=bias)


class ELGNetwork(Module):
    def __init__(self, HG_num=3, input_shape=(1, 96, 160), output_shape=(18, 96, 160), feature_channels=64, cal_look_vec=False):
        super(ELGNetwork, self).__init__()
        self.HG_num = HG_num
        self.cal_look_vec = cal_look_vec
        self.res_before = ResidualBlock(in_channels=input_shape[0], out_channels=feature_channels)

        for i in range(HG_num):
            setattr(self, 'HG_layer_{}'.format(i), HGBlock(feature_channels=feature_channels, layer_num=4))

        if self.cal_look_vec:
            # 用来预测gaze vector
            self.res_after = GazeEstimator(feat_shape=(64, 96, 160))
        else:
            # 用来预测keypoint 的feature map
            self.res_after = ResidualBlock(in_channels=feature_channels, out_channels=output_shape[0])

    def forward(self, x):
        x = self.res_before(x)
        for i in range(self.HG_num):
            x = getattr(self, 'HG_layer_{}'.format(i))(x)
        x = self.res_after(x)
        return x


class EyeLandmarkNetwork(Module):
    def __init__(self, HG_num=3, input_shape=(1, 96, 160), output_shape=(18, 96, 160), feature_channels=64):
        super(EyeLandmarkNetwork, self).__init__()
        self.HG_num = HG_num
        self.res_before = ResidualBlock(in_channels=input_shape[0], out_channels=feature_channels)

        for i in range(HG_num):
            setattr(self, 'HG_layer_{}'.format(i), HGBlock(feature_channels=feature_channels, layer_num=4))

        # 用来预测keypoint 的feature map
        self.res_after = ResidualBlock(in_channels=feature_channels, out_channels=output_shape[0])

    def forward(self, x):
        x = self.res_before(x)
        for i in range(self.HG_num):
            x = getattr(self, 'HG_layer_{}'.format(i))(x)

        x = self.res_after(x)
        return x


class EyeGazeNetwork(Module):
    def __init__(self, HG_num=3, input_shape=(1, 96, 160), feature_channels=64):
        super(EyeGazeNetwork, self).__init__()
        self.HG_num = HG_num
        self.res_before = ResidualBlock(in_channels=input_shape[0], out_channels=feature_channels)

        for i in range(HG_num):
            setattr(self, 'HG_layer_{}'.format(i), HGBlock(feature_channels=feature_channels, layer_num=4))

        self.res_after = GazeEstimator(feat_shape=(64, 96, 160))

    def forward(self, x):
        x = self.res_before(x)
        for i in range(self.HG_num):
            x = getattr(self, 'HG_layer_{}'.format(i))(x)

        x = self.res_after(x)
        return x


class FaceLandmarkNetwork(Module):
    def __init__(self, HG_num=3, input_shape=(), feature_channels=64):
        super(FaceLandmarkNetwork, self).__init__()
        pass
    def forward(self, x):
        pass



class HGBlock(Module): # 根据ELG的论文中的结构来的
    def __init__(self, feature_channels=64, layer_num=4):
        """
        :param feature_channels: 在HGNet里面传播的feature map的通道数
        :param layer_num:  每一个HGNet的一块的downsample的次数
        """
        super(HGBlock, self).__init__()
        self.feature_channels = feature_channels
        self.layer_num = layer_num
        # down sample part
        for i in range(layer_num):
            setattr(self, 'down_res_{}'.format(i), ResidualBlock(feature_channels, feature_channels))
            setattr(self, 'down_pool_{}'.format(i), MaxPool2d(kernel_size=(2, 2)))
        # parallel part
        for i in range(3):
            setattr(self, 'paral_res_{}'.format(i), ResidualBlock(feature_channels, feature_channels))
        # up sample part
        for i in range(layer_num):
            setattr(self, 'up_res_{}'.format(i), ResidualBlock(feature_channels, feature_channels))
            setattr(self, 'up_sample_{}'.format(i), Upsample(scale_factor=2))
        # short cut part
        self.shortcut_conv = Conv2d(in_channels=feature_channels, out_channels=feature_channels, kernel_size=(1, 1))
        # 在forward 函数里面实现
    def forward(self, x):
        cache_stack = []  # 用来保存前面的值，short cut
        # down sample part
        for i in range(self.layer_num):
            x = getattr(self, 'down_res_{}'.format(i))(x)
            x = getattr(self, 'down_pool_{}'.format(i))(x)
            cache_stack.append(x)
        # parallel part
        for i in range(3):
            x = getattr(self, 'paral_res_{}'.format(i))(x)
        # up sample part
        for i in range(self.layer_num):
            residual = cache_stack.pop()
            residual = self.shortcut_conv(residual)
            x_in = residual + x
            x = getattr(self, 'up_res_{}'.format(i))(x_in)
            x = getattr(self, 'up_sample_{}'.format(i))(x)
        return x


class ResidualBlock(Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        # 卷积模块
        half_out_channels = max(int(out_channels/2), 1)
        self.convPart = Sequential(
            Conv2d(in_channels=in_channels, out_channels=half_out_channels, kernel_size=1),
            BatchNorm2d(half_out_channels),
            ReLU(),

            Conv2d(in_channels=half_out_channels, out_channels=int(half_out_channels),
                   kernel_size=3, padding=1),
            BatchNorm2d(half_out_channels),
            ReLU(),

            Conv2d(in_channels=half_out_channels, out_channels=out_channels, kernel_size=1),
            BatchNorm2d(out_channels),
            ReLU()
        )
        # short cut
        # 如果输入输出的层数不同的话那么就应该，通过一个1×1卷积让他们的通道数一样
        if in_channels != out_channels:
            self.skipConv = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.in_channels, self.out_channels = in_channels, out_channels

    def forward(self, x):
        residual = x
        x = self.convPart(x)
        if self.in_channels != self.out_channels:
            residual = self.skipConv(residual)
        x += residual
        return x


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


class GazeEstimator(Module):
    """
    直接根据HGnet 生成的feature map 来预测look vector
    """
    # todo 改成非固定的kernel size
    def __init__(self, feat_shape=(64, 96, 160)):
        super(GazeEstimator, self).__init__()
        self.feat_shape = feat_shape

        self.downsample_block = Sequential(
            ResidualBlock(in_channels=64, out_channels=64),
            MaxPool2d(kernel_size=(2, 2)), # 64 × 48 × 80
            ResidualBlock(in_channels=64, out_channels=32),
            MaxPool2d(kernel_size=(2, 2)), # 32 * 24 * 40
            ResidualBlock(in_channels=32, out_channels=16),
            MaxPool2d(kernel_size=(2, 2)), # 16 * 12 * 20
            ResidualBlock(in_channels=16, out_channels=3)
        )
        # 最后输出的每一个channel 代表一个向量的维度
        self.glob_average_pool = AvgPool2d(kernel_size=(12, 20)) # 3 × 1 × 1

    def forward(self, x):
        x = self.downsample_block.forward(x)
        out = self.glob_average_pool.forward(x)
        return out




def test_residual():
    dummy_input = np.random.randn(32, 1, 96, 160).astype(np.float32)
    dummy_input = torch.from_numpy(dummy_input)
    rs = ResidualBlock(in_channels=1, out_channels=1)
    output = rs.forward(dummy_input)
    print(output.size())


def test_HGnet():
    dummy_input = np.random.randn(32, 64, 96, 160).astype(np.float32)
    dummy_input = torch.from_numpy(dummy_input)
    net = HGBlock(feature_channels=64, layer_num=4)
    output = net.forward(dummy_input)
    print(output.size())


def test_ELGNetwork():
    device = 'cuda:0'
    dummy_input = np.random.randn(1, 1, 96, 160).astype(np.float32)
    dummy_inputs = [dummy_input for i in range(3)]
    net = ELGNetwork().to(device)
    for x in dummy_inputs:
        output = net(torch.from_numpy(x).to(device))
        print(output.size())

        del output


def test_GazeEstimator():
    dummy_input = np.random.randn(32, 64, 96, 160).astype(np.float32)
    dummy_input = torch.from_numpy(dummy_input)
    net = GazeEstimator()
    output = net.forward(dummy_input)
    print(output.size())


def test_ELG_Look_vec():

    device = 'cuda:0'
    dummy_input = np.random.randn(16, 1, 96, 160).astype(np.float32)
    dummy_inputs = [dummy_input for i in range(3)]
    net = ELGNetwork(cal_look_vec=True).to(device)
    for x in dummy_inputs:
        output = net(torch.from_numpy(x).to(device))
        print(output.size())

        del output


if __name__=="__main__":
    # test_residual()
    # test_HGnet()
    # test_ELGNetwork(is_train=False)
    # test_GazeEstimator()
    test_ELG_Look_vec()