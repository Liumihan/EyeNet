import cv2
import numpy as np
from config import opt
from torchvision import transforms
from data.dataset import UnityEyeDataset
from data.transforms import ToTensor, CropEye
from data.utils import get_points_from_heatmaps, draw_points, draw_gaze


def visualize_sample(sample, net, vis, title):
    """
    生成一组visdom中显示的images
    :param sample:
    :return: tuple(np.ndarray, np.ndarray), 两张图片前一张是有标记的， 后一张是原图
    """
    gt_pts = sample['ldmks']
    image_tensor = sample['image']
    # 格式转换与创建不同内存下的副本
    image_bgr_numpy = image_tensor.to('cpu').numpy().transpose(1, 2, 0)
    image_rgb_numpy = cv2.cvtColor(image_bgr_numpy, cv2.COLOR_BGR2RGB)
    pred_dotted_image = image_rgb_numpy.copy()
    gt_dotted_image = image_rgb_numpy.copy()
    # 因为opencv 是BGR 而 visdom 是RGB 并且显示时是C H W
    vis.image(image_rgb_numpy.transpose(2, 0, 1), win='ori_img' + title, opts={'title': title})

    net.eval()
    pred_heatmaps = net.forward(image_tensor.unsqueeze(0).to(opt.device))[-1]
    pred_heatmaps_numpy = pred_heatmaps.cpu().detach().numpy()
    pred_heatmaps_numpy = pred_heatmaps_numpy.squeeze()

    pred_points = get_points_from_heatmaps(pred_heatmaps_numpy) * opt.downsample_scale
    pred_dotted_image = draw_points(pred_dotted_image, pred_points)
    vis.image(pred_dotted_image.transpose(2, 0, 1), win='pred_dotted_img' + title, opts={'title': title + '_pre'})

    gt_dotted_image = draw_points(gt_dotted_image, gt_pts)
    vis.image(gt_dotted_image.transpose(2, 0, 1), win='gt_dotted_img' + title, opts={'title': title + '_gt'})


def visualize_sample_gaze(sample, net, vis, title):
    """
    生成一组visdom中显示的images
    :param sample:
    :return: tuple(np.ndarray, np.ndarray), 两张图片前一张是有标记的， 后一张是原图
    """
    gt_gaze = sample['pitchyaw']
    image_tensor = sample['image']
    # 格式转换与创建不同内存下的副本
    image_bgr_numpy = image_tensor.to('cpu').numpy().transpose(1, 2, 0)
    image_rgb_numpy = cv2.cvtColor(image_bgr_numpy, cv2.COLOR_BGR2RGB)
    pred_dotted_image = image_rgb_numpy.copy()
    gt_dotted_image = image_rgb_numpy.copy()
    # 因为opencv 是BGR 而 visdom 是RGB 并且显示时是C H W
    vis.image(image_rgb_numpy.transpose(2, 0, 1), win='ori_img' + title, opts={'title': title})

    net.eval()
    pred_gaze = net.forward(image_tensor.unsqueeze(0).to(opt.device))
    pred_gaze = pred_gaze.cpu().detach().numpy()
    pred_gaze = pred_gaze.squeeze()

    draw_gaze(pred_dotted_image, pred_gaze)
    vis.image(pred_dotted_image.transpose(2, 0, 1), win='pred_dotted_img' + title, opts={'title': title + '_pre'})

    draw_gaze(gt_dotted_image, gt_gaze)
    vis.image(gt_dotted_image.transpose(2, 0, 1), win='gt_dotted_img' + title, opts={'title': title + '_gt'})


def vis_lines(vis, lines):
    """
    可视化所有的训练数据,比如loss
    :param: lines (dict), {"line_name": line(x, y)} 其中x 和 y 都是(np.NdArray) shape为(1, 1)
    :return: void
    """
    for line_name, num in lines.items():
        vis.line(Y=num[1], X=num[0], win=line_name, update='append' if num[0][0]!= 0 else None, opts=dict(title=line_name))
