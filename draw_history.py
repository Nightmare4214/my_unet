#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os

import matplotlib.pylab as plt
import pandas as pd

if __name__ == '__main__':
    path = os.path.join('history', 'Adam', 'history_RMS3.csv')
    csv_file = pd.read_csv(path)
    print(csv_file.columns)

    plt.figure(1)
    plt.plot(csv_file['epoch'], csv_file['train loss'], label='train loss')
    plt.plot(csv_file['epoch'], csv_file['val loss'], label='val loss')
    plt.legend()

    plt.figure(2)
    plt.plot(csv_file['epoch'], csv_file['train acc'], label='train acc')
    plt.plot(csv_file['epoch'], csv_file['val acc'], label='val acc')
    plt.legend()
    plt.show()

