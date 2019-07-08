import os
import cv2
import json
import torch
import numpy as np
from glob import glob
from config import opt
import scipy.io as sio
from torchvision import transforms
from torch.utils.data import Dataset
from collections.abc import Iterable
from data.utils import draw_points, draw_gaze
from data.utils import vector_to_pitchyaw, pitchyaw_to_vector

class UnityEyeDataset(Dataset):
    def __init__(self, data_dir, eye_size=(160, 96), transform=None, sub_set=None):
        """
        :param data_dir: str, path to the data directory
        :param eye_size: tuple, (W, H) the size of the eye image_bgr which will be
                                sent to the network
        """
        data_dir = os.path.abspath(data_dir)
        self.json_filenames = glob(data_dir + "/*.json")
        if isinstance(sub_set, int):
            self.json_filenames = self.json_filenames[:sub_set]
        elif isinstance(sub_set, Iterable):
            self.json_filenames = self.json_filenames[sub_set[0]: sub_set[1]]
        self.eye_size = eye_size
        self.transform = transform

    def __len__(self):
        return len(self.json_filenames)

    def __getitem__(self, idx):
        json_filename = self.json_filenames[idx]
        img_filename = json_filename[:-5]+".jpg"

        ldmks, look_vec, pitchyaw = self._get_landmarks_and_gaze(json_filename)
        img = cv2.imread(img_filename)

        sample = {'ldmks': ldmks, 'image': img, 'look_vec': look_vec, 'pitchyaw': pitchyaw}

        if self.transform:
            sample = self.transform(sample)

        return sample


    def _get_landmarks_and_gaze(self, json_filename):
        img_filename = json_filename[:-5]+".jpg"
        img = cv2.imread(img_filename)
        data_file = open(json_filename)
        data = json.load(data_file)

        def process_json_list(json_list):
            ldmks = [eval(s) for s in json_list]
            return np.array([(x, img.shape[0] - y, z) for (x, y, z) in ldmks])

        # 获取关键点的数据
        ldmks_iris = process_json_list(data['iris_2d'])[::4, :2]  # 8
        ldmks_iris_center = np.mean(ldmks_iris, axis=0)[:2]  # 1
        ldmks_caruncle = process_json_list(data['caruncle_2d'])[:, :2]  # 7
        ldmks_caruncle_center = np.mean(ldmks_caruncle, axis=0)[:2]
        ldmks_interior_margin = process_json_list(data['interior_margin_2d'])[::2, :2]  # 8
        ldmks = np.vstack((ldmks_iris_center, ldmks_caruncle_center, ldmks_iris, ldmks_interior_margin))

        # 获取注视方向信息
        look_vec = np.array(eval(data['eye_details']['look_vec']))[:3]
        look_vec[1] = -look_vec[1]  # y轴相反

        pitchyaw = vector_to_pitchyaw(look_vec)

        return ldmks, look_vec, pitchyaw


class MPIIGazeDataset(Dataset):
    def __init__(self, img_dir, txt_filepath, eye_size=(160, 96), transform=None):
        self.eye_size = eye_size
        self.transform = transform
        self.img_dir = img_dir
        self.txt_path = txt_filepath
        self._process_txt()

    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, idx):
        filename = self.filename_list[idx]
        pose_vector, look_vector = self.label_list[idx]
        gaze = vector_to_pitchyaw(look_vector.reshape(1, 3))
        img_fullpath = os.path.join(self.img_dir, filename)
        image = cv2.imread(img_fullpath, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, self.eye_size)

        sample = {'image_bgr': image, 'pose_vector': pose_vector, 'look_vec': look_vector, 'gaze': gaze}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def _process_txt(self):
        """
        生成 self.filename_list和self.label_list之后用在getitem里面
        其中labellist的每一个元素是一个tuple(pose_vector, gaze_vector)
        """
        txt_file = open(self.txt_path, 'r')
        data_frame = txt_file.readlines()
        self.filename_list = []
        self.label_list =[]
        for data in data_frame:
            data_str_list = data.strip().split(' [')
            filename = data_str_list[0]
            pose_vector = np.fromstring(data_str_list[1].strip(']'), sep=',')
            gaze_vector = np.fromstring(data_str_list[2].strip(']'), sep=',')
            self.filename_list.append(filename)
            self.label_list.append((pose_vector, gaze_vector))


class LP300WDataset(Dataset):
    def __init__(self, data_dir, mode='3d',tsf=None):
        '''
        :param data_dir: 300wlp的数据集文件夹的路径
        :param mode: 2d 关键点模式还是3d 关键点模式，默认是3d
        :param tsf: transforms
        '''
        self.data_dir = data_dir
        self.mode = mode
        self.tsf = tsf

        self.mat_path_list, self.img_path_list = self._build_path_list()

    def __len__(self):
        return len(self.mat_path_list)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path_list[idx])
        mat = sio.loadmat(self.mat_path_list[idx])

        if self.mode == '3d':
            pts = mat['pts_3d']
        else:
            pts = mat['pts_2d']

        roi_img, pts = self._get_roi_img(img, pts)

        sample = {'image_bgr': roi_img, 'pts': pts,}

        if self.tsf:
            sample = self.tsf(sample)

        return sample

    def _build_path_list(self):

        mat_path_list = []
        img_path_list = []
        landmarks_dir = os.path.join(self.data_dir, 'landmarks/')
        sub_dir_list = os.listdir(landmarks_dir)
        for sub_dir in sub_dir_list:
            sub_dir = os.path.join(landmarks_dir, sub_dir)
            sub_mat_filename_list = os.listdir(sub_dir)
            for mat_filename in sub_mat_filename_list:
                mat_path, img_path = self._transform_matfilename(mat_filename)
                mat_path_list.append(mat_path)
                img_path_list.append(img_path)

        return mat_path_list, img_path_list


    def _transform_matfilename(self, mat_filename):
        """
        将 mat_filename 转换成 mat_path 和 img_path
        :param mat_filename:
        :return:
        """
        mat_path = os.path.join(self.data_dir, 'landmarks/', mat_filename.split('_')[0], mat_filename)
        img_path = os.path.join(self.data_dir, mat_filename.split('_')[0], mat_filename)
        img_path = img_path[:-8] + '.jpg'

        return (mat_path, img_path)

    def _get_roi_img(self, img, pts):
        """
        通过关键点的数据得到roi区域，并且调整pt2d和pt3d的值
        :param pts: 关键点数据
        :return: array： xmin, ymin, xmax, ymax
        """
        roi_x0_ori = np.min(pts, axis=0).astype(int)
        roi_x1_ori = np.max(pts, axis=0).astype(int)

        w = roi_x1_ori[0] - roi_x0_ori[0]
        h = roi_x1_ori[1] - roi_x0_ori[1]

        roi = np.vstack((roi_x0_ori, roi_x1_ori))
        roi_center = np.mean(roi, axis=0).astype(int)

        if w > h:
            s = w
        else:
            s = h
        roi_x0_new = roi_center - int(s / 2 * 1.3)
        roi_x1_new = roi_center + int(s / 2 * 1.3)

        roi_img = img[roi_x0_new[1]:roi_x1_new[1], roi_x0_new[0]:roi_x1_new[0], :]

        pts[:, 0] -= roi_x0_new[0]
        pts[:, 1] -= roi_x0_new[1]

        return roi_img, pts


def visualize_UintyEyesdataset(dataset = UnityEyeDataset(data_dir=opt.train_data_dir)):
    print(len(dataset))
    for sample in dataset:
        image = sample["image"]
        ldmks = sample['ldmks']
        look_vec = sample['look_vec']
        pitchyaw = sample['pitchyaw']
        image_dotted = image.copy()
        image_dotted = draw_points(image_dotted, ldmks)

        # 画gaze 的方向
        iris_center = ldmks[0]
        cal_look_vec = pitchyaw_to_vector(pitchyaw)
        cal_pitchyaw = vector_to_pitchyaw(cal_look_vec)
        draw_gaze(image_dotted, pitchyaw, iris_center)
        cv2.imshow('img_dotted', image_dotted)
        print('T[{:.2f}, {:.2f}] C[{:.2f}, {:.2f}]'.format(pitchyaw[0], pitchyaw[1], cal_pitchyaw[0], cal_pitchyaw[1]))

        if cv2.waitKey(0) & 0xff == ord("q"):
            break
    cv2.destroyAllWindows()


# def visualize_MPIIGaze(dataset = MPIIGazeDataset(
#     img_dir='/media/liumihan/HDD_Documents/眼部数据集/MPIIGaze/Data/Normalized/imgs', txt_filepath='/media/liumihan/HDD_Documents/眼部数据集/MPIIGaze/Data/Normalized/all.txt')):
#     l = len(dataset)
#     for i, sample in enumerate(dataset):
#         img = sample['image_bgr']
#         look_vec = sample['look_vec']
#         cv2.arrowedLine(img, pt1=(80, 48), pt2=(int(80+80*look_vec[0]), int(48+80*look_vec[1])), color=(255, 255, 255), thickness=2)
#         cv2.imshow('img', img)
#         print('{} / {}'.format(i+1, l))
#         if cv2.waitKey(1) & 0xff == ord('q'):
#             break
#     cv2.destroyAllWindows()


# def test_transform():
#     # ToTensor
#     # dataset = UnityEyeDataset("/home/liumihan/Desktop/DSM-FinalDesign/驾驶数据集/imgs", transform=ToTensor())
#     # ZeroMean
#     dataset = UnityEyeDataset("/home/liumihan/Desktop/DSM-FinalDesign/驾驶数据集/imgs", transform=ZeroMean())
#     # Compose
#     dataset = UnityEyeDataset("/home/liumihan/Desktop/DSM-FinalDesign/驾驶数据集/imgs",
#                               transform=transforms.Compose([ZeroMean(), ToTensor()]))
#     for sample in dataset:
#         a = sample


def visualize_LP300wDataset():
    dataset = LP300WDataset(data_dir='/media/liumihan/HDD_Documents/Datesets/300W-LP/300W_LP')
    for sample in dataset:
        img = sample['image_bgr']
        pts = sample['pts']

        for p in pts:
            cv2.circle(img, center=(p[0], p[1]), radius=1, color=(255, 255, 255), thickness=2)

        cv2.imshow('img', img)

        if cv2.waitKey(0) & 0xff == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    visualize_UintyEyesdataset()
    # test_transform()
    # img = cv2.imread('/media/liumihan/HDD_Documents/眼部数据集/MPIIGaze/Data/Normalized/imgs/p00day21_left1.jpg')
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # visualize_MPIIGaze(dataset= MPIIGazeDataset(img_dir=opt.MPIIGaze_img_dir, txt_filepath=opt.MPIIGaze_train_txt))
    # visualize_LP300wDataset()