import math
import torch
import visdom
import numpy as np
from config import opt
from torch import optim
from model import EyeNet_ldmk
from torch.nn import MSELoss
from vis import visualize_sample
from torchvision import transforms
from torch.utils.data import DataLoader
from data.dataset import UnityEyeDataset
from data.transforms import ToTensor, CropEye
from data.utils import draw_gaussian, get_points_from_heatmaps


def train():

    vis = visdom.Visdom(env='EyeNet_0.1', port=11223)
    tsf = transforms.Compose([CropEye(shaking=True), ToTensor()])

    datasets = {phase: UnityEyeDataset(data_dir=opt.data_dir, transform=tsf, sub_set=getattr(opt, '{}_subset'.format(phase)))
                for phase in ['train', 'val']}
    dataloaders = {phase: DataLoader(dataset=datasets[phase] , batch_size=opt.batch_size, shuffle=True, num_workers=4)
                   for phase in ['train', 'val']}

    net = EyeNet_ldmk()

    start_epoch = 0
    best_epoch_loss = 10000
    if opt.checkpoint_path:
        checkpoint = torch.load(opt.checkpoint_path, map_location=opt.device)
        net.load_state_dict(checkpoint['net_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_epoch_loss = checkpoint['val_epoch_loss']

    net.float()
    net.to(opt.device)
    criterion = MSELoss()
    optimizer = optim.RMSprop(net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    for epoch in range(start_epoch, opt.epochs, 1):
        train_epoch_loss = 0.0
        val_epoch_loss = 0.0
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
                sample_num = len(datasets[phase])
                iter_per_epoch = math.ceil(sample_num / opt.batch_size)
                for itr, batch in enumerate(dataloaders[phase]):
                    t_vmin, t_vmax, t_euler_diff, t_iter_loss = train_phase(batch, optimizer, net, criterion)
                    train_epoch_loss = (train_epoch_loss * itr + t_iter_loss.item()) / (itr + 1)

                    print('[Epoch {} / {}, iter {} / {}] train_loss: {:.6f} max: {:.4f} min: {:.4f}, euler distance:{:.1f}'.format(
                        epoch, opt.epochs, itr * opt.batch_size, sample_num, t_iter_loss.item(), t_vmax, t_vmin, t_euler_diff
                    ))
                    # # 可视化
                    # 绘制各种线条
                    vis.line(Y=np.array([t_iter_loss.item()]), X=np.array([(epoch * iter_per_epoch) + (itr + 1)]),
                             win='Train_loss',
                             update='append' if (epoch + 1) * (itr + 1) > 0 else None, opts=dict(title='train_loss'))
                    vis.line(Y=np.array([t_vmax.item()]), X=np.array([(epoch * iter_per_epoch) + (itr + 1)]),
                             win='t_v_max',
                             update='append' if (epoch + 1) * (itr + 1) != 0 else None, opts=dict(title='t_v_max'))
                    vis.line(Y=np.array([t_vmin.item()]), X=np.array([(epoch * iter_per_epoch) + (itr + 1)]),
                             win="t_v_min",
                             update='append' if (epoch + 1) * (itr + 1) != 0 else None, opts=dict(title='t_v_min'))
                    vis.line(Y=np.array([t_euler_diff]), X=np.array([(epoch * iter_per_epoch) + (itr + 1)]),
                             win="t_euler_distance",
                             update='append' if (epoch + 1) * (itr + 1) != 0 else None,
                             opts=dict(title='t_euler_distance'))
                    # # 图片显示
                    if itr % opt.plot_every_iter == 0:
                        random_idx = np.random.randint(0, len(datasets[phase]))
                        vis_sample = datasets[phase][random_idx]
                        visualize_sample(vis_sample, net, vis, title='train_sample')
            elif phase == 'val':
                sample_num = len(datasets[phase])
                iter_per_epoch = math.ceil(sample_num / opt.batch_size)
                net.eval()
                for itr, batch in enumerate(dataloaders[phase]):
                    v_vmin, v_vmax, v_euler_diff, v_iter_loss = val_phase(batch, net, criterion)
                    val_epoch_loss = (val_epoch_loss * itr + v_iter_loss.item()) / (itr + 1)

                    print('[Epoch {} / {}, iter {} / {}] val_loss: {:.6f} max: {:.4f} min: {:.4f}, euler distance:{:.1f}'.format(
                        epoch, opt.epochs, itr * opt.batch_size, sample_num, v_iter_loss.item(), v_vmax, v_vmin, v_euler_diff
                    ))
                    # # 可视化
                    # 绘制各种线条
                    vis.line(Y=np.array([v_iter_loss.item()]), X=np.array([(epoch * iter_per_epoch) + (itr + 1)]),
                             win='val_loss',
                             update='append' if (epoch + 1) * (itr + 1) > 0 else None, opts=dict(title='val_loss'))
                    vis.line(Y=np.array([v_vmax.item()]), X=np.array([(epoch * iter_per_epoch) + (itr + 1)]),
                             win='v_v_max',
                             update='append' if (epoch + 1) * (itr + 1) != 0 else None, opts=dict(title='v_v_max'))
                    vis.line(Y=np.array([v_vmin.item()]), X=np.array([(epoch * iter_per_epoch) + (itr + 1)]),
                             win="v_v_min",
                             update='append' if (epoch + 1) * (itr + 1) != 0 else None, opts=dict(title='v_v_min'))
                    vis.line(Y=np.array([v_euler_diff]), X=np.array([(epoch * iter_per_epoch) + (itr + 1)]),
                             win="v_euler_distance",
                             update='append' if (epoch + 1) * (itr + 1) != 0 else None,
                             opts=dict(title='v_euler_distance'))

                    if itr % opt.plot_every_iter == 0:
                        random_idx = np.random.randint(0, len(datasets[phase]))
                        vis_sample = datasets[phase][random_idx]
                        visualize_sample(vis_sample, net, vis, title='val_sample')

        vis.line(Y=np.array([train_epoch_loss, val_epoch_loss]).reshape(1, 2),
                 X=np.array([epoch, epoch]).reshape(1, 2),
                 win='epoch_loss', update='append' if epoch > 0 else None,
                 opts={'title': 'epoch loss', 'legend': ['train_phase loss', 'val_phase loss']})

        print('epoch_loss: {:.5f} old_best_epoch_loss: {:.5f}'.format(train_epoch_loss, best_epoch_loss))
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


def train_phase(batch, optimizer, net, criterion):
    #  训练一个iter
    net.train()

    inputs = batch["image"].to(opt.device)
    pts = batch['ldmks'].numpy()

    optimizer.zero_grad()

    heatmaps_pred = net.forward(inputs)[-1]  # 因为在网络里面每一个HG我都保留了，输出

    # 生成heatmap_targets
    B, C, H, W = heatmaps_pred.size()
    heatmap_targets = torch.zeros_like(heatmaps_pred, device='cpu').numpy()
    for b in range(B):
        for c in range(C):
            downsample_scale = opt.downsample_scale
            pt = pts[b, c] / downsample_scale
            draw_gaussian(image=heatmap_targets[b, c], point=pt, size=opt.gau_size)
    heatmap_targets = torch.from_numpy(heatmap_targets).to(opt.device)

    # 多loss 回传, 效果不好放弃使用了
    loss = criterion(heatmap_targets, heatmaps_pred)
    loss.backward()
    optimizer.step()
    # 预测点与真实点之间的欧拉距离
    heatmaps_pred_numpy = heatmaps_pred.cpu().detach().numpy()
    total_distance = 0.0
    for i, heatmaps in enumerate(heatmaps_pred_numpy):
        points_pred = get_points_from_heatmaps(heatmaps)
        diff = pts[i] - points_pred * opt.downsample_scale
        pow = np.power(diff, 2)
        summ = np.sum(pow)
        distence = math.sqrt(summ)
        total_distance += distence
    batch_mean_distance = total_distance / opt.batch_size

    # 统计最大值和最小值
    v_max = torch.max(heatmaps_pred)
    v_min = torch.min(heatmaps_pred)

    return v_min, v_max, batch_mean_distance, loss


def val_phase(batch, net, criterion):
    #  验证一个iter
    net.eval()
    inputs = batch["image"].to(opt.device)
    pts = batch['ldmks'].numpy()

    heatmaps_pred = net.forward(inputs)[-1]  # 因为在网络里面每一个HG我都保留了，输出
    # 生成heatmap_targets
    B, C, H, W = heatmaps_pred.size()
    heatmap_targets = torch.zeros_like(heatmaps_pred, device='cpu').numpy()
    for b in range(B):
        for c in range(C):
            downsample_scale = opt.downsample_scale
            pt = pts[b, c] / downsample_scale
            draw_gaussian(image=heatmap_targets[b, c], point=pt, size=opt.gau_size)
    heatmap_targets = torch.from_numpy(heatmap_targets).to(opt.device)

    # 多loss 回传, 效果不好放弃使用了
    loss = criterion(heatmap_targets, heatmaps_pred)
    # 预测点与真实点之间的欧拉距离
    heatmaps_pred_numpy = heatmaps_pred.cpu().detach().numpy()
    total_distance = 0.0
    for i, heatmaps in enumerate(heatmaps_pred_numpy):
        points_pred = get_points_from_heatmaps(heatmaps)
        diff = pts[i] - points_pred * 4
        pow = np.power(diff, 2)
        summ = np.sum(pow)
        distance = math.sqrt(summ)
        total_distance += distance
    batch_mean_distance = total_distance / opt.batch_size

    # 统计最大值和最小值
    v_max = torch.max(heatmaps_pred)
    v_min = torch.min(heatmaps_pred)

    return v_min, v_max, batch_mean_distance, loss


if __name__ == '__main__':
    # from data.utils import find_wrong_imgs
    # find_wrong_imgs(opt.data_dir)
    train()
