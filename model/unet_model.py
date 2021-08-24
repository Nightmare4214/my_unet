#!/usr/bin/env python
# _*_ coding:utf-8 _*_
from model.unet_parts import *


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super(UNet, self).__init__()
        self.conv_down1 = ConvDown(in_channels, 64)
        self.conv_down2 = ConvDown(64, 128)
        self.conv_down3 = ConvDown(128, 256)
        self.conv_down4 = ConvDown(256, 512)
        self.conv_down5 = ConvDown(512, 1024)
        # self.dropout = nn.Dropout(p=0.5)
        self.conv_up1 = ConvUP(1024, 512)
        self.conv_up2 = ConvUP(512, 256)
        self.conv_up3 = ConvUP(256, 128)
        self.conv_up4 = ConvUP(128, 64)

        self.conv_out = nn.Conv2d(64, out_channels, 1, stride=1, padding=0)

    def forward(self, x):
        x, conv1 = self.conv_down1(x)
        x, conv2 = self.conv_down2(x)
        x, conv3 = self.conv_down3(x)
        x, conv4 = self.conv_down4(x)
        _, x = self.conv_down5(x)
        # x = self.dropout(x)
        x = self.conv_up1(x, conv4)
        x = self.conv_up2(x, conv3)
        x = self.conv_up3(x, conv2)
        x = self.conv_up4(x, conv1)
        x = self.conv_out(x)

        return x


if __name__ == '__main__':
    im = torch.randn(1, 1, 572, 572)
    model = UNet(in_channels=1, out_channels=2)
    print(model)
    x = model(im)
    print(x.shape)
