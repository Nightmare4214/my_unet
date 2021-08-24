#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import random

import numpy
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates


def elastic_transform(image, alpha, sigma, seed=None):
    """
    弹性形变

    :param image: 图片(h,w)
    :param alpha: 放缩因子
    :param sigma: 弹性系数
    :param seed: 随机种子
    :return: 弹性形变后的图片
    """
    assert isinstance(image, numpy.ndarray)

    shape = image.shape  # h*w
    assert 2 == len(shape)
    if seed is None:
        seed = random.randint(1, 100)
    random_state = np.random.RandomState(seed)
    # 生成一个均匀分布(-1,1)的移位场,然后高斯滤波，然后成缩放
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    # 生成坐标
    y, x = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    # 偏移
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
    # 插值
    return map_coordinates(image, indices, order=1).reshape(shape), seed


def padding_elastic_transform(image, alpha, sigma, seed=None, pad_size=20):
    image_size = image.shape[0]
    image = np.pad(image, pad_size, mode="symmetric")
    image, seed = elastic_transform(image, alpha=alpha, sigma=sigma, seed=seed)
    return crop(image, image_size, pad_size, pad_size), seed


def image_add_value(image, value):
    """
    图片+一个值

    :param image: 图片
    :param value: 值
    :return: 处理后的图片
    """
    # 增加有可能超出图片范围，要先转类型，然后限制到255，再转回去
    return np.clip(image.astype('int16') + value, 0, 255).astype('uint8')


def add_gaussian_noise(image, mean, std):
    gauss_noise = np.random.normal(mean, std, image.shape)
    return image_add_value(image, gauss_noise)


def add_uniform_noise(image, low, high):
    uniform_noise = np.random.uniform(low, high, image.shape)
    return image_add_value(uniform_noise, uniform_noise)


def change_brightness(image, value):
    """
    增加图片亮度

    :param image: 图片
    :param value: 增加亮度
    :return: 调亮的图片
    """
    return image_add_value(image, value)


def crop(image, crop_size, height_crop_start, width_crop_start):
    """
    图像切割（正方形）

    :param image: 图像(h,w)
    :param crop_size: 切割大小
    :param height_crop_start: h方向上裁剪位置
    :param width_crop_start: w方向上裁剪位置
    :return: 切割后的图片
    """
    return image[height_crop_start:height_crop_start + crop_size,
           width_crop_start:width_crop_start + crop_size]


def stride_size(image_size, crop_num, crop_size):
    """
    计算切割图片的步长

    :param image_size: 图片长度
    :param crop_num: 切割数量
    :param crop_size: 切割长度
    :return: 步长
    """

    # 有重叠，要保证最后一块切完是刚好 (crop_num-1)crop_size+crop_size=image_size
    return (image_size - crop_size) // (crop_num - 1)


def multi_cropping(image, crop_size, crop_num1, crop_num2):
    """
    图像切割成左上，右上，左下，右下

    :param image: 图片
    :param crop_size: 切割大小
    :param crop_num1: h切割数量
    :param crop_num2: w切割数量
    :return: [左上，右上，左下，右下]
    """
    img_height, img_width = image.shape[0], image.shape[1]
    # 要能够切完整个图片
    assert crop_size * crop_num1 >= img_width and crop_size * crop_num2 >= img_height
    # 不能切太多
    assert crop_num1 <= img_width - crop_size + 1 and crop_num2 <= img_height - crop_size + 1

    cropped_images = []
    height_stride = stride_size(img_height, crop_num1, crop_size)
    width_stride = stride_size(img_width, crop_num2, crop_size)

    for i in range(crop_num1):
        for j in range(crop_num2):
            cropped_images.append(crop(image, crop_size, height_stride * i, width_stride * j))

    return np.asarray(cropped_images)
