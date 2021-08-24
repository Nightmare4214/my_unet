#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import glob
import os

import cv2
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torchvision import transforms

from utils.preprocess import *


class ValidDataset(Dataset):
    def __init__(self, image_path, mask_path, in_size=572, out_size=388) -> None:
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.images = glob.glob(os.path.join(image_path, '*'))
        self.images.sort()

        if mask_path:
            self.masks = glob.glob(os.path.join(mask_path, '*'))
            self.masks.sort()
        else:
            self.masks = None
        self.data_len = len(self.images)
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0, 1)
        ])

    def __getitem__(self, index) -> T_co:
        image = cv2.imread(self.images[index], 0)

        pad_size = (self.in_size - self.out_size) // 2
        # 为了更好预测边缘，使用镜像padding
        image = np.pad(image, pad_size, mode='symmetric')
        # 切割成左上，右上，左下，右下
        cropped_images = multi_cropping(image,
                                        crop_size=self.in_size,
                                        crop_num1=2, crop_num2=2)
        processed_list = np.empty(cropped_images.shape, dtype=np.float32)
        for i in range(len(cropped_images)):
            processed_list[i] = self.image_transform(cropped_images[i])
        cropped_images = torch.from_numpy(processed_list)
        if self.masks:
            mask = cv2.imread(self.masks[index], 0)
            cropped_masks = multi_cropping(mask,
                                           crop_size=self.out_size,
                                           crop_num1=2, crop_num2=2)
            mask = torch.from_numpy(mask / 255).long()
            cropped_masks = torch.from_numpy(cropped_masks / 255).long()
        else:
            mask, cropped_masks = None, None

        # torch.Size([4, 572, 572]),torch.Size([4, 388, 388]),torch.Size([512, 512])
        return cropped_images, cropped_masks, mask

    def __len__(self):
        return self.data_len


if __name__ == '__main__':
    valid_dataset = ValidDataset(r'..\data\val\images',
                                 r'..\data\val\masks')
    cropped_images, cropped_masks, mask = valid_dataset.__getitem__(0)
    print(cropped_images)
    print(cropped_masks)
    print(mask)
    print(cropped_images.shape)
    print(cropped_masks.shape)
    print(mask.shape)
    print(cropped_images.dtype)
    print(cropped_masks.dtype)
    print(mask.dtype)
