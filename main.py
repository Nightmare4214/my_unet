#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import csv
import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset.train_dataset import TrainDataset
from dataset.valid_dataset import ValidDataset
from model.unet_model import UNet
from utils.SoftDiceLoss import SoftDiceLoss
from utils.util import get_device, get_prediction_image, save_image

use_wandb = True  # 使用wandb
device = get_device()  # gpu
use_weight = False  # 带权重的交叉熵
use_cross_entropy = True  # 使用交叉熵
use_dice_loss = True  # 使用dice
epoch_start = 0
epoch_end = 2000
lr = 0.0001  # 学习率
ce_lambda = 1.0  # 交叉熵系数
dice_lambda = 1.0  # dice系数
batch_size = 6  # 批次大小
cur_time = time.time()


def get_loss(outputs, masks, criterion, dice_loss=None):
    loss = torch.tensor(0.0).to(device)
    if criterion:
        loss += ce_lambda * criterion(outputs, masks)
    if dice_loss:
        loss += dice_lambda * dice_loss(outputs, masks)
    return loss


def train_model(model, train_data_loader, criterion, optimizer, dice_loss=None):
    """
    训练模型

    :param model: 模型
    :param train_data_loader: 训练集
    :param criterion: 损失
    :param optimizer: 优化器
    :return: 损失，准确率
    """
    model.train()
    total_acc = 0
    total_loss = 0
    batch = 0
    for images, masks in train_data_loader:
        batch += 1
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images)
        loss = get_loss(outputs, masks, criterion, dice_loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            predict = torch.argmax(outputs, dim=1).float()
            batch_size, height, width = masks.size()
            acc = 1.0 * torch.eq(predict, masks).sum().item() / (batch_size * height * width)
            total_acc += acc
            total_loss += loss.cpu().item()
    return total_acc / batch, total_loss / batch


def validate_model(model, valid_data_loader, criterion, save_dir, dice_loss=None):
    """
    验证模型（batch_size=1)

    :param model: 模型
    :param valid_data_loader: 验证集
    :param criterion: 损失
    :param save_dir: 保存图片
    :return: 损失，准确率
    """
    model.eval()
    total_acc = 0
    total_loss = 0
    batch = 0
    cnt = 0
    batch_size = 1
    os.makedirs(save_dir, exist_ok=True)
    for cropped_image, cropped_mask, origin_mask in valid_data_loader:
        # 1*4*572*572 1*4*388*388 1*512*512
        batch += 1
        with torch.no_grad():
            # 用来存储4个切割
            stacked_image = torch.Tensor([]).to(device)  # 4*388*388
            for i in range(cropped_image.size(1)):
                images = cropped_image[:, i, :, :].unsqueeze(0).to(device)  # 1*1*572*572
                masks = cropped_mask[:, i, :, :].to(device)  # 1*388*388
                outputs = model(images)  # 1*388*388
                loss = get_loss(outputs, masks, criterion, dice_loss)
                predict = torch.argmax(outputs, dim=1).float()
                total_loss += loss.cpu().item()
                stacked_image = torch.cat((stacked_image, predict))
            origin_mask = origin_mask.to(device)
            for j in range(batch_size):
                cnt += 1
                predict_image = get_prediction_image(stacked_image)
                save_image(predict_image, os.path.join(save_dir, f'{cnt}.bmp'))
                batch_size, height, width = origin_mask.size()
                # predict_image = predict_image.unsqueeze(0)
                acc = 1.0 * torch.eq(predict_image, origin_mask).sum().item() / (batch_size * height * width)
                total_acc += acc
    return total_acc / batch, total_loss / (batch * 4)


def save_model(model, path, epoch, optimizer):
    path = os.path.join(path, f'epoch_{epoch}')
    os.makedirs(path, exist_ok=True)
    state = {'model': model.state_dict(), 'epoch': epoch, 'optimizer': optimizer.state_dict()}
    torch.save(state, os.path.join(path, f"model_epoch_{epoch}.pth"))


if __name__ == '__main__':
    hyper_param_map = {
        'use_weight': use_weight,
        'use_cross_entropy': use_cross_entropy,
        'use_dice_loss': use_dice_loss,
        'lr': lr,
        'epoch': epoch_end,
        'batch_size': batch_size,
        'ce_lambda': ce_lambda,
        'dice_lambda': dice_lambda
    }
    train_image_path = os.path.join('data', 'train', 'images')
    train_mask_path = os.path.join('data', 'train', 'masks')
    valid_image_path = os.path.join('data', 'val', 'images')
    valid_mask_path = os.path.join('data', 'val', 'masks')

    train_dataset = TrainDataset(train_image_path, train_mask_path)
    valid_dataset = ValidDataset(valid_image_path, valid_mask_path)

    train_data_loader = DataLoader(train_dataset, num_workers=10, batch_size=batch_size, shuffle=True)
    # 为了方便写，这里batch_size必须为1
    valid_data_loader = DataLoader(valid_dataset, num_workers=3, batch_size=1, shuffle=False)

    model = UNet(in_channels=1, out_channels=2).to(device)
    weight = torch.Tensor([2, 1]).to(device) if use_weight else None
    criterion = nn.CrossEntropyLoss(weight) if use_cross_entropy else None
    dice_loss = SoftDiceLoss() if use_dice_loss else None
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.99)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    header = ['epoch', 'train loss', 'train acc', 'val loss', 'val acc']
    history_path = os.path.join('history', str(cur_time))
    model_save_dir = os.path.join(history_path, 'saved_models')
    image_save_path = os.path.join(history_path, 'result_images')
    save_file_name = os.path.join(history_path, 'history.csv')
    os.makedirs(history_path, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(image_save_path, exist_ok=True)
    if use_wandb:
        import wandb
        wandb.init(project="unet",
                   name='unet_' + str(cur_time))
        wandb.config.update(hyper_param_map)
        wandb.define_metric("train/*")
        wandb.define_metric("valid/*")
        wandb.watch(model)
    else:
        with open(save_file_name, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)
        with open(os.path.join(history_path, 'hyper_param.txt'), 'w') as f:
            f.write(str(hyper_param_map))

    print("Initializing Training!")
    for i in range(epoch_start, epoch_end):
        train_acc, train_loss = train_model(model, train_data_loader, criterion, optimizer, dice_loss)

        print('Epoch', str(i + 1), 'Train loss:', train_loss, "Train acc", train_acc)
        if (i + 1) % 5 == 0:
            val_acc, val_loss = validate_model(
                model, valid_data_loader, criterion, os.path.join(image_save_path, f'epoch{i + 1}'), dice_loss)
            print('Val loss:', val_loss, "val acc:", val_acc)
            if use_wandb:
                wandb.log({
                    'train/acc': train_acc,
                    'train/loss': train_loss,
                    'valid/acc': val_acc,
                    'valid/loss': val_loss
                }, step=i + 1)
            else:
                values = [i + 1, train_loss, train_acc, val_loss, val_acc]
                with open(save_file_name, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(values)

            if (i + 1) % 10 == 0:
                save_model(model, model_save_dir, i + 1, optimizer)
