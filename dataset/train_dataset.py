#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import glob
import os
import random

import cv2
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torchvision import transforms

from utils.preprocess import *


class TrainDataset(Dataset):
    def __init__(self, image_path, mask_path, in_size=572, out_size=388) -> None:
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.images = glob.glob(os.path.join(image_path, '*'))
        self.masks = glob.glob(os.path.join(mask_path, '*'))
        self.images.sort()
        self.masks.sort()
        self.data_len = len(self.images)
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0, 1)
        ])

    def __getitem__(self, index) -> T_co:
        image = cv2.imread(self.images[index], 0)
        mask = cv2.imread(self.masks[index], 0)

        # 翻转
        flip_choice = random.randint(-1, 2)
        if flip_choice != 2:
            image = cv2.flip(image, flip_choice)
            mask = cv2.flip(mask, flip_choice)

        # 添加噪声
        if random.randint(0, 1):
            image = add_gaussian_noise(image, 0, random.randint(0, 20))
        else:
            low, high = random.randint(-20, 0), random.randint(0, 20)
            image = add_uniform_noise(image, low, high)

        # 调整亮度
        brightness = random.randint(-20, 20)
        image = change_brightness(image, brightness)

        # 弹性形变
        sigma = random.randint(6, 12)
        image, seed = padding_elastic_transform(image, alpha=34, sigma=sigma, seed=None, pad_size=20)

        mask, _ = padding_elastic_transform(mask, alpha=34, sigma=sigma, seed=seed, pad_size=20)
        # mask只有0和255，所以需要二值化
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        h, w = image.shape
        pad_size = (self.in_size - self.out_size) // 2
        # 为了更好预测边缘，使用镜像padding
        image = np.pad(image, pad_size, mode='symmetric')
        height_crop_start = random.randint(0, h - self.out_size)
        width_crop_start = random.randint(0, w - self.out_size)
        # 对应论文中，预测黄色的部分需要将蓝色部分输入
        image = crop(image, crop_size=self.in_size, height_crop_start=height_crop_start,
                     width_crop_start=width_crop_start)
        mask = crop(mask, crop_size=self.out_size, height_crop_start=height_crop_start,
                    width_crop_start=width_crop_start)

        image = self.image_transform(image)
        mask = torch.from_numpy(mask / 255).long()

        # torch.Size([1, 572, 572]),torch.Size([388, 388])
        return image, mask

    def __len__(self):
        return self.data_len


if __name__ == '__main__':
    train_dataset = TrainDataset(r'..\data\train\images',
                                 r'..\data\train\masks')

    image, mask = train_dataset.__getitem__(0)
    print(image)
    print(mask)
    print(image.shape)
    print(mask.shape)
    print(image.dtype)
    print(mask.dtype)
