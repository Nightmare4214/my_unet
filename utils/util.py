#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import cv2
import torch

from utils.preprocess import stride_size


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_image(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为cv2格式

    :param input_tensor: 要保存的tensor(h*w)
    :param filename: 保存的文件名
    """
    # assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    if input_tensor.dtype == torch.long:
        input_tensor = input_tensor.float()

    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 从[0,1]转化为[0,255]，最后转为cv2
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).type(torch.uint8).numpy()
    cv2.imwrite(filename, input_tensor)


def get_weight_mat(crop_size, crop_num1, crop_num2, img_height, img_width):
    """
    计算重叠次数

    :param crop_size: 切割大小
    :param crop_num1: h切割数
    :param crop_num2: w切割数
    :param img_height: 图片h
    :param img_width: 图片w
    :return: 重叠次数
    """

    # 最终结果
    res = torch.zeros((img_height, img_width))
    # 与切割大小相同的全1矩阵
    one_mat = torch.ones((crop_size, crop_size))
    # 步长
    height_stride = stride_size(img_height, crop_num1, crop_size)
    width_stride = stride_size(img_width, crop_num2, crop_size)
    for i in range(crop_num1):
        for j in range(crop_num2):
            res[height_stride * i:height_stride * i + crop_size,
                width_stride * j:width_stride * j + crop_size] += one_mat
    return res


def image_concatenate(image, crop_num1, crop_num2, img_height, img_width):
    """
    切割图片拼接

    :param image: 切割图片（4*388*388）
    :param crop_num1: h切割数
    :param crop_num2: w切割数
    :param img_height: 图片h
    :param img_width: 图片w
    :return: 拼接图片
    """
    # 切割大小
    crop_size = image.size(2)
    # 最终结果
    res = torch.zeros((img_height, img_width)).to(get_device())
    # 步长
    height_stride = stride_size(img_height, crop_num1, crop_size)
    width_stride = stride_size(img_width, crop_num2, crop_size)
    cnt = 0
    for i in range(crop_num1):
        for j in range(crop_num2):
            res[height_stride * i:height_stride * i + crop_size,
                width_stride * j:width_stride * j + crop_size] += image[cnt]
            cnt += 1
    return res


def get_prediction_image(stacked_img):
    """
    预测图片

    :param stacked_img: 切割的图片（4*388*388)
    :return: 预测图片
    """
    # 计算重叠次数
    div_arr = get_weight_mat(388, 2, 2, 512, 512).to(get_device())
    # 拼接图片
    img_concat = image_concatenate(stacked_img, 2, 2, 512, 512)
    # 因为有重叠，所以取平均
    return img_concat/div_arr
