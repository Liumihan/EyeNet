import cv2
import torch
import numpy as np
from config import opt
from data.dataset import UnityEyeDataset
from data.utils import draw_points, draw_gaussian


class ToTensor(object):

    def __call__(self, sample):

        image = sample["image"]
        gaze = sample["pitchyaw"]

        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image.astype(np.float32))
        gaze = torch.from_numpy(gaze.astype(np.float32))

        sample["image"] = image
        sample["pitchyaw"] = gaze

        return sample


class CropEye(object):

    def __init__(self, size=(160, 96), shaking=True):
        self.size = size
        self.shaking = shaking

    def __call__(self, sample):
        image = sample["image"]
        ldmks = sample['ldmks']
        # 对眼睛区域进行裁剪
        # 1.根据GazeML里面的处理方法
        left_corner = ldmks[1]
        right_corner = ldmks[14]
        eye_width = 2.0 * abs(left_corner[0]- right_corner[0])
        eye_height = self.size[1] / self.size[0] * eye_width
        eye_middle = np.mean((np.min(ldmks[10:18], axis=0),
                              np.max(ldmks[10:18], axis=0)), axis=0)
        # 随机上下左右抖动
        if self.shaking:
            dx = np.random.randint(-int(eye_width * 0.2), int(eye_width * 0.2))
            dy = np.random.randint(-int(eye_height * 0.2), int(eye_height * 0.2))
            eye_middle = eye_middle - np.array([dx, dy])
        # 2.从中心向两边寻找裁剪的边界
        lowx = np.max(eye_middle[0] - eye_width / 2, 0)
        lowy = np.max(eye_middle[1] - eye_height / 2, 0)
        highx = min(eye_middle[0] + eye_width / 2, image.shape[1])
        highy = min(eye_middle[1] + eye_height / 2, image.shape[0])

        # 3.关键点坐标相应的减掉
        ldmks = ldmks - np.array([lowx, lowy])

        # 4. crop
        image = image[int(lowy):int(highy+1), int(lowx):int(highx+1), :]

        # 对图像进行缩放
        scale_h = self.size[1] / image.shape[0]
        scale_w = self.size[0] / image.shape[1]
        ldmks = ldmks * np.array([scale_w, scale_h])
        image = cv2.resize(image, self.size)
        sample['image'] = image
        sample['ldmks'] = ldmks

        return sample



class Blur(object):
    def __init__(self):
        pass
    def __call__(self, *args, **kwargs):
        pass


class HistogramEqual(object):

    def __init__(self):
        pass
    def __call__(self, *args, **kwargs):
        pass


class RGBNoise(object):

    def __init__(self, difficulty):
        self.difficulty = difficulty

    def __call__(self, *args, **kwargs):
        pass


class LineNoise(object):
    def __init__(self):
        pass
    def __call__(self, *args, **kwargs):
        pass

def test_totensor():
    dataset = UnityEyeDataset(data_dir=opt.train_data_dir, transform=ToTensor())
    for sample in dataset:
        image = sample['image']
        print(image.size())

def test_cropeye():
    dataset = UnityEyeDataset(data_dir=opt.train_data_dir, transform=CropEye())
    for sample in dataset:
        image = sample['image']
        ldmks = sample['ldmks']
        image = draw_points(image, ldmks)
        cv2.imshow('image', image)

        if cv2.waitKey(0) == ord('q'):
            break
    cv2.destroyAllWindows()


def test_draw_gaussin():
    dataset = UnityEyeDataset(data_dir=opt.train_data_dir, transform=CropEye())
    for sample in dataset:
        image = sample['image']
        ldmks = sample['ldmks']
        image = draw_points(image, ldmks)
        cv2.imshow('image', image)
        heatmap = np.zeros(shape=(96, 160))

        for i, ldmk in enumerate(ldmks):
            draw_gaussian(heatmap, ldmk, size=7)
            cv2.imshow('heatmap', heatmap)

        if cv2.waitKey(0) == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # test_totensor()
    # test_cropeye()
    test_draw_gaussin()