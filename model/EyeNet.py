import torch
import numpy as np
from tensorboardX import SummaryWriter
# from vis.utils import get_points_from_heatmaps
from model.utils import *
from data.utils import get_points_from_heatmaps
from thop import profile  # 统计模型的参数量和计算量
from torchsummary import summary

class EyeNet(Module):
    def __init__(self):
        super(EyeNet, self).__init__()

    def load(self, checkpoint_path, device='cuda:0'):
        checkpoint = torch.load(f=checkpoint_path, map_location=device)
        self.load_state_dict(checkpoint['net_state_dict'])

    def predict(self, image, device='cuda:0'):
        """
        输入一张RGB图像, 输出一组tensor
        :param image: (np.array) 输入的图像, shape是(H, W, 3)
        :param device: (str) 采用的设备, 默认是cuda:0
        :return: output: (torch.Tensor) 输出的张量, size由网络的决定
        """

        assert image.shape[2] == 3, '输入的图片必须是RGB图像'
        tensor = torch.from_numpy(image.astype(np.float32))
        tensor = tensor.permute(2, 0, 1)
        tensor = tensor.unsqueeze(0)
        output = self.forward(tensor.to(device))

        return output


class EyeNet_ldmk(EyeNet):
    def __init__(self, hg_num=3, input_channels=3, num_classes = 18, feature_channels=128):
        super(EyeNet_ldmk, self).__init__()
        assert  feature_channels % 4 == 0, 'feature channels 必须要能被4整除'
        self.hg_num = hg_num
        self.in_channels = input_channels
        self.classes = num_classes
        # 因为一般眼部图像的分辨率都较小所以没有添加7x7的卷积层
        self.res_before = Sequential(
            Conv2d(self.in_channels, int(feature_channels/4), kernel_size=3, stride=1, padding=1),
            BatchNorm2d(int(feature_channels/4)),
            ReLU(True),
            ResBlock(in_channels=int(feature_channels/4), out_channels=int(feature_channels/2)),
            ResBlock(in_channels=int(feature_channels/2), out_channels=int(feature_channels))
        )
        for i in range(hg_num-1):
            setattr(self, 'HG_expand_layer_{}'.format(i), HGBlock_expand(feature_channels, num_classes=self.classes))
        self.last_hg = HGBlock(feature_channels, depth=4)

        self.look_vector_predictor = Sequential(ResBlock(feature_channels, feature_channels),
                                      Conv2d(feature_channels, feature_channels, 1, 1),
                                      BatchNorm2d(feature_channels),
                                      ReLU(True),
                                      Conv2d(feature_channels, self.classes, 1, 1))

    def forward(self, x):
        x = self.res_before(x)
        outputs = []
        for i in range(self.hg_num - 1):
            temp_output, x = getattr(self, 'HG_expand_layer_{}'.format(i))(x)
            outputs.append(temp_output)
        x = self.last_hg(x)
        output = self.look_vector_predictor(x)
        # outputs.append(output)
        # 也就是说outputs[-1]是真正的网络的输出
        return output

    def predict(self, image, device='cuda:0'):
        """
        输入一张RGB图像, 输出一组预测的关键点坐标
        :param image: (np.array) 输入的图像, shape为(H, W, 3)
        :param device: (str) 将要采用的设备, 默认是cuda:0
        :return: pred_points: (np.array) 输出的关键点坐标, shape是(18, 2)
        """

        pred_heatmaps = super(EyeNet_ldmk, self).predict(image, device)
        pred_heatmap = pred_heatmaps[-1]
        pred_heatmap = pred_heatmap.detach().cpu().numpy()
        pred_points = get_points_from_heatmaps(pred_heatmap.squeeze())

        return pred_points

class EyeNet_gaze(EyeNet):
    def __init__(self, hg_num=3, input_channels=3, num_classes=2, feature_channels=64):  # class 代表pitch , yaw
        super(EyeNet_gaze, self).__init__()
        assert  feature_channels % 4 == 0, 'feature channels 必须要能被4整除'
        self.hg_num = hg_num
        self.in_channels = input_channels
        self.classes = num_classes
        # 因为一般眼部图像的分辨率都较小所以没有添加7x7的卷积层
        self.res_before = Sequential(
            Conv2d(self.in_channels, int(feature_channels/4), kernel_size=7, stride=2, padding=3),
            BatchNorm2d(int(feature_channels/4)),
            ReLU(True),
            ResBlock(in_channels=int(feature_channels/4), out_channels=int(feature_channels/2)),
            ResBlock(in_channels=int(feature_channels/2), out_channels=int(feature_channels))
        )
        for i in range(hg_num-1):
            setattr(self, 'HG_expand_layer_{}'.format(i), HGBlock_expand(feature_channels, num_classes=self.classes))
        self.last_hg = HGBlock(feature_channels, depth=4)

        # self.look_vector_predictor = Sequential(
        #     ResBlock(feature_channels, feature_channels),
        #     DownSampleConvBlock(feature_channels, int(feature_channels/2)),  # 48 x 80
        #     DownSampleConvBlock(int(feature_channels/2), int(feature_channels/4)),  # 24 x 40
        #     DownSampleConvBlock(int(feature_channels/4), int(feature_channels/8)),# 12 x 20
        #     Conv2d(in_channels=int(feature_channels/8), out_channels=self.classes, kernel_size=1, stride=1))
        self.gaze_predictor = GazeEstimatBlock(feature_channels, num_classes)

    def forward(self, x):

        x = self.res_before(x)
        for i in range(self.hg_num - 1):
            temp_output, x = getattr(self, 'HG_expand_layer_{}'.format(i))(x)
        x = self.last_hg(x)
        x = self.gaze_predictor(x)
        # h, w = x.size(2), x.size(3)
        # output = F.avg_pool2d(x, kernel_size=(h, w))
        # output = output.squeeze(-1).squeeze(-1)
        output = x

        return output


def test_eyenet_ldmk():
    net = EyeNet_ldmk(hg_num=3)
    dummy_input = torch.randn(size=(1, 3, 96, 160))
    output = net.forward(dummy_input)
    print(output.size())
    flops, params = profile(model=net, inputs=(dummy_input, ))
    print("FLOPS: {}, Params:{} ".format(flops, params))

    summary(net.to('cuda:0'), input_size=(3, 96, 160))

    net.to('cpu')
    with SummaryWriter(comment='Eyenet_ldmks-0') as w:
        w.add_graph(net, dummy_input)


def test_eyenet_gaze():
    net = EyeNet_gaze(hg_num=3)
    dummy_input = torch.randn(size=(1, 3, 96, 160))
    output = net.forward(dummy_input)
    print(output.size())
    flops, params = profile(model=net, inputs=(dummy_input, ))
    print("FLOPS: {}, Params:{} ".format(flops, params))

    summary(net.to('cuda:0'), input_size=(3, 96, 160))
    net.to('cpu')
    with SummaryWriter(comment='Eyenet_gaze-0') as w:
        w.add_graph(net, dummy_input)


if __name__ == '__main__':
    test_eyenet_gaze()
    # test_eyenet_ldmk()




