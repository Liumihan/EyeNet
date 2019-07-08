import cv2
import torch
import numpy as np
from config import opt
from model.EyeNet import EyeNet_ldmk
from data.dataset import UnityEyeDataset
from data.transforms import CropEye, ToTensor
from data.utils import draw_points, get_points_from_heatmaps

if __name__ == '__main__':
    dataset = UnityEyeDataset(data_dir=opt.dev_data_dir)
    net = EyeNet_ldmk().cuda()
    net.eval()
    checkpoint_path = '/home/liumihan/Desktop/DSM-FinalDesign/code/Reference/GazeNet/weights/EyeNet-0.1_epoch0.pth'
    net.load(checkpoint_path)
    for sample in dataset:
        sample = CropEye(shaking=0)(sample)
        image_ori = sample['image'].copy()
        cv2.imshow('image_ori', image_ori)

        pred_points = net.predict(image_ori)
        image = draw_points(image_ori, pred_points)

        cv2.imshow('image', image)
        if cv2.waitKey(0) == ord('q'):
            break
    cv2.destroyAllWindows()