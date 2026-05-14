"""
 @Time    : 2021/7/6 14:52
 @Author  : Haiyang Mei
 @E-mail  : mhy666@mail.dlut.edu.cn

 @Project : CVPR2021_PFNet
 @File    : train.py
 @Function: Training

"""
import datetime
import time
import os

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm

import joint_transforms
from config import cod_training_root
from config import backbone_path
from datasets import ImageFolder
from misc import AvgMeter, check_mkdir
from PFNet import PFNet

import loss

cudnn.benchmark = True

torch.manual_seed(2021)
device_ids = [0]

ckpt_path = './ckpt'
exp_name = 'PFNet'

args = {
    'epoch_num': 45,
    'train_batch_size': 16,
    'last_epoch': 0,
    'lr': 1e-3,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': '',
    'scale': 416,
    'save_point': [],
    'poly_train': True,
    'optimizer': 'SGD',
    # edge auxiliary learning (multi-task)
    'edge_kernel_size': 3,
    # 边缘损失权重（最终上限），配合 edge_warmup_epoch 做逐步引入
    'edge_loss_weight': 0.5,
    # 边缘损失从 0 线性 warm-up 到 edge_loss_weight 所需的 epoch 数
    'edge_warmup_epoch': 10,
}

print(torch.__version__)

# Path.
check_mkdir(ckpt_path)
check_mkdir(os.path.join(ckpt_path, exp_name))
vis_path = os.path.join(ckpt_path, exp_name, 'log')
check_mkdir(vis_path)
log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')
writer = SummaryWriter(log_dir=vis_path, comment=exp_name)

# Transform Data.
joint_transform = joint_transforms.Compose([
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.Resize((args['scale'], args['scale']))
])
img_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()

# Prepare Data Set.
train_set = ImageFolder(cod_training_root, joint_transform, img_transform, target_transform)
print("Train set: {}".format(train_set.__len__()))
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=16, shuffle=True)

total_epoch = args['epoch_num'] * len(train_loader)

# loss function
structure_loss = loss.structure_loss().cuda(device_ids[0])
bce_loss = nn.BCEWithLogitsLoss().cuda(device_ids[0])
iou_loss = loss.IOU().cuda(device_ids[0])

def bce_iou_loss(pred, target):
    bce_out = bce_loss(pred, target)
    iou_out = iou_loss(pred, target)

    loss = bce_out + iou_out

    return loss

def mask_to_boundary(mask, kernel_size=3, thresh=0.5):
    """
    由二值/软 mask 生成边缘（boundary/edge）监督信号。
    参考思路：形态学梯度（dilate - erode），常用于 Boundary-Guided 系列方法的边缘监督。

    mask: (B,1,H,W) in [0,1]
    return: (B,1,H,W) in [0,1]
    """
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1
    pad = kernel_size // 2

    m = (mask > thresh).float()
    # dilation: max pooling
    dilate = torch.nn.functional.max_pool2d(m, kernel_size=kernel_size, stride=1, padding=pad)
    # erosion: min pooling implemented by max pooling on inverted mask
    erode = 1.0 - torch.nn.functional.max_pool2d(1.0 - m, kernel_size=kernel_size, stride=1, padding=pad)
    boundary = (dilate - erode).clamp(0.0, 1.0)
    return boundary

def dice_loss_with_logits(logits, target, eps=1e-6):
    """
    Dice loss for edge supervision (handle extreme class imbalance).
    logits: (B,1,H,W)
    target: (B,1,H,W) in {0,1}
    """
    prob = torch.sigmoid(logits)
    prob = prob.view(prob.size(0), -1)
    target = target.view(target.size(0), -1)
    inter = (prob * target).sum(dim=1)
    union = prob.sum(dim=1) + target.sum(dim=1)
    dice = (2 * inter + eps) / (union + eps)
    return (1 - dice).mean()

def main():
    print(args)
    print(exp_name)

    net = PFNet(backbone_path).cuda(device_ids[0]).train()

    if args['optimizer'] == 'Adam':
        print("Adam")
        optimizer = optim.Adam([
            {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
             'lr': 2 * args['lr']},
            {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
             'lr': 1 * args['lr'], 'weight_decay': args['weight_decay']}
        ])
    else:
        print("SGD")
        optimizer = optim.SGD([
            {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
             'lr': 2 * args['lr']},
            {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
             'lr': 1 * args['lr'], 'weight_decay': args['weight_decay']}
        ], momentum=args['momentum'])

    if len(args['snapshot']) > 0:
        print('Training Resumes From \'%s\'' % args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
        total_epoch = (args['epoch_num'] - int(args['snapshot'])) * len(train_loader)
        print(total_epoch)

    net = nn.DataParallel(net, device_ids=device_ids)
    print("Using {} GPU(s) to Train.".format(len(device_ids)))

    open(log_path, 'w').write(str(args) + '\n\n')
    train(net, optimizer)
    writer.close()

def train(net, optimizer):
    curr_iter = 1
    start_time = time.time()

    for epoch in range(args['last_epoch'] + 1, args['last_epoch'] + 1 + args['epoch_num']):
        loss_record, loss_1_record, loss_2_record, loss_3_record, loss_4_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

        train_iterator = tqdm(train_loader, total=len(train_loader))
        for data in train_iterator:
            if args['poly_train']:
                base_lr = args['lr'] * (1 - float(curr_iter) / float(total_epoch)) ** args['lr_decay']
                optimizer.param_groups[0]['lr'] = 2 * base_lr
                optimizer.param_groups[1]['lr'] = 1 * base_lr

            inputs, labels = data
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda(device_ids[0])
            labels = Variable(labels).cuda(device_ids[0])

            optimizer.zero_grad()

            predict_1, predict_2, predict_3, predict_4, edge_pred = net(inputs)

            loss_1 = bce_iou_loss(predict_1, labels)
            loss_2 = structure_loss(predict_2, labels)
            loss_3 = structure_loss(predict_3, labels)
            loss_4 = structure_loss(predict_4, labels)

            edge_gt = mask_to_boundary(labels, kernel_size=args['edge_kernel_size'])
            # 结合 Dice + BCE 让边缘监督更稳定（Dice 解决不平衡，BCE 稳定概率）
            edge_loss_dice = dice_loss_with_logits(edge_pred, edge_gt)
            edge_loss_bce = bce_loss(edge_pred, edge_gt)
            edge_loss = 0.5 * edge_loss_dice + 0.5 * edge_loss_bce

            # 边缘损失采用 epoch 级别 warm-up，前若干个 epoch 主要保留原 PFNet 行为
            if args['edge_warmup_epoch'] > 0:
                edge_weight = args['edge_loss_weight'] * max(
                    0.0, min(1.0, float(epoch) / float(args['edge_warmup_epoch']))
                )
            else:
                edge_weight = args['edge_loss_weight']

            loss = 1 * loss_1 + 1 * loss_2 + 2 * loss_3 + 4 * loss_4 + edge_weight * edge_loss

            loss.backward()

            optimizer.step()

            loss_record.update(loss.data, batch_size)
            loss_1_record.update(loss_1.data, batch_size)
            loss_2_record.update(loss_2.data, batch_size)
            loss_3_record.update(loss_3.data, batch_size)
            loss_4_record.update(loss_4.data, batch_size)

            if curr_iter % 10 == 0:
                writer.add_scalar('loss', loss, curr_iter)
                writer.add_scalar('loss_1', loss_1, curr_iter)
                writer.add_scalar('loss_2', loss_2, curr_iter)
                writer.add_scalar('loss_3', loss_3, curr_iter)
                writer.add_scalar('loss_4', loss_4, curr_iter)
                writer.add_scalar('loss_edge', edge_loss, curr_iter)

            log = '[%3d], [%6d], [%.6f], [%.5f], [%.5f], [%.5f], [%.5f], [%.5f]' % \
                  (epoch, curr_iter, base_lr, loss_record.avg, loss_1_record.avg, loss_2_record.avg,
                   loss_3_record.avg, loss_4_record.avg)
            train_iterator.set_description(log)
            open(log_path, 'a').write(log + '\n')

            curr_iter += 1

        if epoch in args['save_point']:
            net.cpu()
            torch.save(net.module.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % epoch))
            net.cuda(device_ids[0])

        if epoch >= args['epoch_num']:
            net.cpu()
            torch.save(net.module.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % epoch))
            print("Total Training Time: {}".format(str(datetime.timedelta(seconds=int(time.time() - start_time)))))
            print(exp_name)
            print("Optimization Have Done!")
            return

if __name__ == '__main__':
    main()