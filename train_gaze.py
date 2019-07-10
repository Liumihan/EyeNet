import math
import copy
import torch
import visdom
import numpy as np
from config import opt
from torch import optim
from model import EyeNet_gaze, EyeNet_ldmk
from model.utils import GazeEstimatBlock
from torch.nn import MSELoss, SmoothL1Loss
from vis import visualize_sample_gaze, vis_lines
from torchvision import transforms
from torch.utils.data import DataLoader
from data.dataset import UnityEyeDataset
from data.transforms import ToTensor, CropEye
from data.utils import draw_gaussian, get_points_from_heatmaps


def train():

    vis = visdom.Visdom(env='EyeNet_gaze-0.1', port=11223)
    tsf = transforms.Compose([CropEye(shaking=False), ToTensor()])

    datasets = {phase: UnityEyeDataset(data_dir=getattr(opt, phase + "_data_dir"), transform=tsf)for phase in ['train', 'val']}
    dataloaders = {phase: DataLoader(dataset=datasets[phase] , batch_size=opt.batch_size, shuffle=True, num_workers=4)
                   for phase in ['train', 'val']}

    # net = EyeNet_gaze()
    # 尝试迁移学习
    net_ldmk = EyeNet_ldmk()
    start_epoch = 0
    best_epoch_loss = 10000
    if opt.checkpoint_path:
        checkpoint = torch.load(opt.checkpoint_path, map_location=opt.device)
        net_ldmk.load_state_dict(checkpoint['net_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        # best_epoch_loss = checkpoint['val_epoch_loss']

    # 将网络的尾部改成gaze_estimateor
    net = copy.deepcopy(net_ldmk)
    num_ftrs = 128  # 这里设置成了固定值
    net.look_vector_predictor = GazeEstimatBlock(num_ftrs, 2)


    net.float()
    net.to(opt.device)
    criterion = MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    for epoch in range(start_epoch, opt.epochs, 1):
        train_epoch_loss = 0.0
        val_epoch_loss = 0.0
        for phase in ['train', 'val']:
            if phase == 'train':
                sample_num = len(datasets[phase])
                iter_per_epoch = math.ceil(sample_num / opt.batch_size)
                for itr, batch in enumerate(dataloaders[phase]):
                    t_iter_loss, t_itr_diff = train_phase(batch, optimizer, net, criterion, opt.device)
                    train_epoch_loss = (train_epoch_loss * itr + t_iter_loss.item()) / (itr + 1)
                    # 打印输出
                    print('[Epoch {} / {}, iter {} / {}] train_loss: {:4.2f} pitch diff:{:4.2f} yaw diff:{:4.2f}'.format(
                          epoch, opt.epochs, itr * opt.batch_size, sample_num, t_iter_loss.item(),
                          abs(t_itr_diff[0]), abs(t_itr_diff[1])))
                    # # 可视化
                    # 绘制各种线条
                    x_value = np.array([(epoch * iter_per_epoch) + (itr + 1)])
                    lines = {"train_loss": (x_value, np.array([t_iter_loss.item()])),
                             "t_pitch_diff": (x_value, np.array([t_itr_diff[0].item()])),
                             "t_yaw_diff": (x_value, np.array([t_itr_diff[1].item()]))}
                    vis_lines(vis, lines)
                    # # 图片显示
                    if itr % opt.plot_every_iter == 0:
                        random_idx = np.random.randint(0, len(datasets[phase]))
                        vis_sample = datasets[phase][random_idx]
                        visualize_sample_gaze(vis_sample, net, vis, title='train_sample')
            elif phase == 'val':
                sample_num = len(datasets[phase])
                iter_per_epoch = math.ceil(sample_num / opt.batch_size)
                for itr, batch in enumerate(dataloaders[phase]):
                    v_itr_loss, v_itr_diff = val_phase(batch, net, criterion)
                    val_epoch_loss = (val_epoch_loss * itr + v_itr_loss.item()) / (itr + 1)

                    print('[Epoch {} / {}, iter {} / {}] val_loss: {:4.2f} pitch diff:{:4.2f} yaw diff:{:4.2f}'.format(
                          epoch, opt.epochs, itr * opt.batch_size, sample_num, v_itr_loss.item(),
                          abs(v_itr_diff[0]), abs(v_itr_diff[1])))
                    # # 可视化
                    # 绘制各种线条
                    x_value = np.array([(epoch * iter_per_epoch) + (itr + 1)])
                    lines = {"v_loss": (x_value, np.array([v_itr_loss.item()])),
                             "v_pitch_diff": (x_value, np.array([v_itr_diff[0].item()])),
                             "v_yaw_diff": (x_value, np.array([v_itr_diff[1].item()]))}
                    vis_lines(vis, lines)
                    if itr % opt.plot_every_iter == 0:
                        random_idx = np.random.randint(0, len(datasets[phase]))
                        vis_sample = datasets[phase][random_idx]
                        visualize_sample_gaze(vis_sample, net, vis, title='val_sample')
        # 绘制每一个epoch的 train loss 和 val Loss
        vis.line(Y=np.array([train_epoch_loss, val_epoch_loss]).reshape(1, 2),
                 X=np.array([epoch, epoch]).reshape(1, 2),
                 win='epoch_loss', update='append' if epoch > 0 else None,
                 opts={'title': 'epoch loss', 'legend': ['train loss', 'val loss']})
        # 是否需要保存模型
        print('epoch_loss: {:.5f} old_best_epoch_loss: {:.5f}'.format(val_epoch_loss, best_epoch_loss))
        if val_epoch_loss < best_epoch_loss:
            print('epoch loss < best_epoch_loss, save these weights.')
            best_epoch_loss = val_epoch_loss
            torch.save({
                'epoch': epoch,
                't_epoch_loss': train_epoch_loss,
                'val_epoch_loss': val_epoch_loss,
                'net_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, opt.weight_save_dir + opt.saving_prefix + '_epoch{}.pth'.format(epoch))


def train_phase(batch, optimizer, net, criterion, device="cuda:0"):
    #  训练一个iter
    net.train()
    optimizer.zero_grad()
    net.zero_grad()

    inputs = batch["image"].to(device)
    gaze_targets = batch["pitchyaw"].to(device)

    gaze_pred = net.forward(inputs)
    loss = criterion(gaze_targets, gaze_pred.squeeze())
    loss.backward()
    optimizer.step()

    diff = gaze_targets - gaze_pred.squeeze(-1).squeeze(-1)
    diff = torch.abs(diff)
    diff = diff.mean(dim=0)

    return loss, diff


def val_phase(batch, net, criterion, device="cuda:0"):
    #  验证一个iter
    net.eval()
    inputs = batch["image"].to(opt.device)
    gaze_targets = batch["pitchyaw"].to(device)

    gaze_pred = net.forward(inputs)

    # 多loss 回传, 效果不好放弃使用了
    loss = criterion(gaze_targets, gaze_pred.squeeze())  # batch_size必须大于1
    # 预测点与真实点之间的欧拉距离
    diff = gaze_targets - gaze_pred.squeeze(-1).squeeze(-1)
    diff = torch.abs(diff)
    diff = diff.mean(dim=0)

    return loss, diff


if __name__ == '__main__':
    train()
