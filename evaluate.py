import cv2
import torch
from tqdm import tqdm
from model import EyeNet_gaze
from data.dataset import MPIIGazeDataset
from data.transforms import ToTensor


def evalutate_on_mpii(save_filename, device='cuda:0'):
    tsf = None
    dataset = MPIIGazeDataset(img_dir="/media/liumihan/HDD_Documents/眼部数据集/MPIIGaze/Data/Normalized/imgs",
                              txt_filepath="/media/liumihan/HDD_Documents/眼部数据集/MPIIGaze/Data/Normalized/left_one_out/p00_leftout/only_p00.txt",
                              tsf=tsf)
    checkpoint_path = "weights/EyeNet_gaze-MPIIGaze_00_epoch0.pth"
    net = EyeNet_gaze()
    net.load(checkpoint_path, device=device)
    net.eval()
    net.to(device)


    counter = 0
    sample_nums = len(dataset)
    f = open("./evaluations/" + save_filename, 'w')
    with torch.no_grad():
        total_diff = torch.zeros(size=(1, 2)).to(device).to(torch.float32)
        for sample in dataset:
            image = sample['image']
            pitchyaw = sample['pitchyaw']
            pitchyaw = torch.from_numpy(pitchyaw).to(device).to(torch.float32)

            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            pred_pitchyaw = net.predict(image, device=device)

            diff = pitchyaw - pred_pitchyaw.squeeze(-1).squeeze(-1)
            diff = torch.abs(diff)
            diff = diff.mean(dim=0)

            total_diff += diff
            counter += 1
            mean_diff = total_diff / counter
            line = "[{} / {}]mean error : pitch {:.3f}, yaw {:.3f}".format(counter, sample_nums, mean_diff[0][0].item(), mean_diff[0][1].item())
            print(line)
            f.writelines(line)

    f.close()


if __name__ == '__main__':
    evalutate_on_mpii("p00_left_out.txt")




