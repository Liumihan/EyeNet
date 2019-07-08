import os
import cv2
import glob
import json
import math
import torch
import pprint
import numpy as np
from config import opt
from tqdm import  tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


point_structure_18 = {'iris': tuple(range(2, 10)) + (2, ),
                      'interior_margin': tuple(range(10, 18)) + (10, ),
                      'iris_center': (0, ),
                      'inner_corner': (1, )}


def find_wrong_imgs(img_dir):
    """
    找出该文件夹下面所有对应错误图片的json
    :param img_dir: str, path to the data directory
    :return: void
    """
    json_filenames = glob.glob(img_dir + "/*.json")
    l = len(json_filenames)
    wrong_img = []
    for json_filename in tqdm(json_filenames):
        delete = False
        try:
            img_filename = json_filename[:-5]+".jpg"
            img = cv2.imread(img_filename)
            data_file = open(json_filename)
            data = json.load(data_file)
        except:
            delete = True

        def process_json_list(json_list):
            ldmks = [eval(s) for s in json_list]
            return np.array([(x, img.shape[0] - y, z) for (x, y, z) in ldmks])
        try:
            # 获取关键点的数据
            ldmks_iris = process_json_list(data['iris_2d'])[::4, :2]  # 8
            ldmks_iris_center = np.mean(ldmks_iris, axis=0)[:2]  # 1
            ldmks_caruncle = process_json_list(data['caruncle_2d'])[:, :2]  # 7
            ldmks_caruncle_center = np.mean(ldmks_caruncle, axis=0)[:2]
            ldmks_interior_margin = process_json_list(data['interior_margin_2d'])[::2, :2]  # 8

            # 获取注视方向信息
            look_vec = np.array(eval(data['eye_details']['look_vec']))[:3]
            look_vec[1] = -look_vec[1]  # y轴相反
        except:
            delete = True

        if len(ldmks_iris) <= 0 or len(ldmks_caruncle) <= 0 or len(ldmks_interior_margin) <= 0:
            delete = True

        if len(look_vec) <= 0:
            delete = True

        if img is None:
            delete = True

        if delete:
            os.popen('rm -f {}'.format(json_filename))
            print('{} has no correspond img, deleted'.format(json_filename))
    print('there are {} wrong imgs'.format(len(wrong_img)))
    pprint.pprint(wrong_img)


# def vector_to_pitchyaw(vectors):
#     r"""Convert given gaze vectors to yaw (:math:`\theta`) and pitch (:math:`\phi`) angles.
#
#     Args:
#         vectors (:obj:`numpy.array`): gaze vectors in 3D :math:`(n\times 3)`.
#
#     Returns:
#         :obj:`numpy.array` of shape :math:`(n\times 2)` with values in radians.
#     """
#     n = vectors.shape[0]
#     out = np.empty((n, 2))
#     vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
#     out[:, 0] = np.arcsin(vectors[:, 1])  # phi pitch
#     out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # theta yaw
#     return out
#
#
# def pitchyaw_to_vector(pitchyaws):
#     r"""Convert given yaw (:math:`\theta`) and pitch (:math:`\phi`) angles to unit gaze vectors.
#
#     Args:
#         pitchyaws (:obj:`numpy.array`): yaw and pitch angles :math:`(n\times 2)` in radians.
#
#     Returns:
#         :obj:`numpy.array` of shape :math:`(n\times 3)` with 3D vectors per row.
#     """
#     n = pitchyaws.shape[0]
#     sin = np.sin(pitchyaws)
#     cos = np.cos(pitchyaws)
#     out = np.empty((n, 3))
#     out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])
#     out[:, 1] = sin[:, 0]
#     out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])
#     return out


# # 按照我自己的标准算的  x 轴向右, y 轴向下, z轴向内, 以z轴负方向为初始方向的pitch和yaw
# https://www.cnblogs.com/graphics/archive/2012/08/10/2627458.html
# 想不通的时候就看看这个图仔细分析一下
def vector_to_pitchyaw(vector):

    vector = vector * 100  # 为了数值的稳定
    pitch = np.arctan2(vector[1], -vector[2]) / np.pi * 180  # 这里的z轴都是负数因为肯定是向外看的，所以加了个负号
    yaw = np.arctan2(vector[0], -vector[2]) / np.pi * 180

    # pitch 向下为正方向, yaw 向右为正方向
    return np.array((pitch, yaw))


# 假定向量的长度为1, x 轴向右, y 轴向下, z轴向内, 以z轴负方向为初始方向的pitch和yaw
def pitchyaw_to_vector(pitchyaw):
    pitch, yaw = pitchyaw
    # 转成弧度
    pitch, yaw = pitch / 180 * np.pi, yaw / 180 * np.pi
    tg_pitch, tg_yaw = math.tan(pitch), math.tan(yaw)
    z = math.sqrt(1 / (tg_pitch ** 2 + tg_yaw ** 2 + 1))
    x = z * tg_yaw
    y = z * tg_pitch

    return np.array((x, y, -z)) # z 取负数的原理同上


def draw_points(image_bgr, points):
    """
    将所给的关键点在图片上显示出来
    :param image_bgr: (np.array) 待显示的图片 (H, W, C) , 应当是opencv格式的bgr图片
    :param points:  (np.array) shape 应当是(68, 2), 是所有68个关键点的集合
    :return: image_bgr: (np.array) 已经被dotted的图像
    """
    idx = 0
    for part, point_idxs in point_structure_18.items():
        part_points = []  # 为了后面画他们的多边形
        for idx in point_idxs:
            x, y = points[idx]
            cv2.circle(image_bgr, center=(int(x), int(y)), radius=1, thickness=2, color=(255, 0, 0))
            # cv2.putText(image_bgr, text=str(idx), org=(int(x), int(y)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.25,
            #             color=(255, 0, 255))
            idx += 1
            part_points.append((int(x), int(y)))
        cv2.polylines(image_bgr, pts=np.array([part_points]), isClosed=False, color=(255, 255, 255))

    return image_bgr


def draw_gaze(image, pitchyaw, start=None):
    if start is not None:
        ox, oy = int(start[0]), int(start[1])
    else:
        ox, oy = int(image.shape[1]/2), int(image.shape[0]/2)

    cal_look_vec = pitchyaw_to_vector(pitchyaw)
    cv2.arrowedLine(image, (ox, oy), (int(ox + 80 * cal_look_vec[0]), int(oy + 80 * cal_look_vec[1])),
                    color=(0, 255, 255), thickness=2)

def _gaussian_distribution(size=3, sigma=0.25, amplitude=1, normalize=False,
              width=None, height=None, sigma_horz=None, sigma_vert=None):
    """
    :param size:
    :param sigma:
    :param mean:
    :param amplitude: (int) 表示这些这个高斯分布的顶峰的高度
    :param normalize:
    :param width:
    :param height:
    :param sigma_horz:
    :param sigma_vert:
    :return:
    """
    # 将一些值取为默认值
    if width is None:
        width = size
    if height is None:
        height = size
    if sigma_horz is None:
        sigma_horz = sigma
    if sigma_vert is None:
        sigma_vert = sigma

    def _gauss(x, y):
        # 根据百度百科两个不相关的二维正态分布, 使ρ为0
        # 我只想要这个形状, 所以前面的常数部分都没有要了， 并且让这个形状更加的圆润，分别除以了 width 和 height
        result = math.exp(- math.pow(x/sigma_horz/width, 2) - math.pow(y/sigma_vert/height, 2))

        return result

    center_x = 0.5 * width + 0.5
    center_y = 0.5 * height + 0.5
    # 先申请gauss的空间
    m_gauss = np.empty((height, width), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            m_gauss[i][j] = amplitude * _gauss(j + 1 - center_x, i + 1 - center_y)

    if normalize:
        m_gauss = m_gauss / np.sum(m_gauss)

    return m_gauss


def draw_gaussian(image, point, size):
    """
    在图片上生成高斯分布
    :param image: (np.array) shape应该是（W, H)
    :param point: (tuple) 将从这个点作为中心生成高斯分布图
    :param size: (int) 想要生成的高斯分布图的大小，一般为 7 9 11 13 等奇数
    :return: none
    """
    assert isinstance(size, int) and (size % 2 == 1), '输入的size必须为7, 9, 11, 13等奇数'
    h, w = image.shape[:2]
    gau = _gaussian_distribution(size=size, sigma=0.25, normalize=False)
    # 判断高斯分布是不是在图像内，如果不是在图像内的话，需要有一些删减
    px, py = point[0], point[1]
    px, py = int(px), int(py)
    gau_half_w = size // 2

    for i in range(size):
        for j in range(size):
            ih = py+i-gau_half_w
            iw = px+j-gau_half_w
            if ih >= 0 and ih < h and iw >=0 and iw < w:
                image[ih, iw] = gau[i, j]


def get_peek_point(heatmap):
    """
    :param heatmap: np.ndarray, (96, 160)
    :return: x, y: int64, int64
    """
    y, x = np.where(heatmap == heatmap.max())
    if len(y) > 0:
        y = y[0]
    if len(x) > 0:
        x = x[0]

    return x, y


def show_3d_eye_ldmks(json_filename):
    img_filename = json_filename[:-5] + ".jpg"
    img = cv2.imread(img_filename)
    data_file = open(json_filename)
    data = json.load(data_file)

    def process_json_list(json_list):
        ldmks = [eval(s) for s in json_list]
        return np.array([(x, img.shape[0] - y, z) for (x, y, z) in ldmks])

    ldmks_iris = process_json_list(data['iris_2d'])  # 8
    ldmks_iris_center = np.mean(ldmks_iris, axis=0)  # 1
    ldmks_caruncle = process_json_list(data['caruncle_2d'])  # 7
    ldmks_caruncle_center = np.mean(ldmks_caruncle, axis=0)
    ldmks_interior_margin = process_json_list(data['interior_margin_2d'])  # 8
    ldmks = np.vstack((ldmks_iris_center, ldmks_caruncle_center, ldmks_iris, ldmks_interior_margin))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    xs = ldmks[:, 0]
    ys = ldmks[:, 1]
    zs = ldmks[:, 2]

    ax.scatter(xs=xs, ys=ys, zs=zs, marker='o')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.grid(False)
    ax.set_xlim(200, 500)
    ax.set_ylim(200, 500)

    plt.show()


def get_points_from_heatmaps(heatmaps):
    """
    从多张热力图中获得关键点集合
    :param heatmaps: (np.array) shape应该是(68, H, W)
    :return: points: (np.array) shape应该是(68, 2)
    """
    L = heatmaps.shape[0]
    points = np.empty(shape=(L, 2))
    for i, heatmap in enumerate(heatmaps):
        x, y = get_peek_point(heatmap)
        points[i][0], points[i][1] = x, y
    return points

if __name__ == "__main__":
    find_wrong_imgs('/home/luoyc/Daihuanhuan/datasets/eye/UnityEyes100w')

    # show_3d_eye_ldmks('/media/liumihan/HDD_Documents/眼部数据集/UnityEyes/val_imgs/83.jpg')
    pass