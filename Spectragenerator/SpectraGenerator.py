#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy import interpolate

count = 0

data_path = './AfterLI/WSe2/'

all_files_path=[] # 获取所有文件的完整路径名
for root, dirs, files in os.walk(data_path,topdown=False):
    if len(files)>0:
        each_foder_files=[os.path.join(root,x) for x in files]
        all_files_path.extend(each_foder_files)

docnum = len(all_files_path) #Number of spectra

for file in all_files_path:
    data = np.loadtxt(file, encoding='utf-8')
    x = data[:,0] # 读取第一列数据
    y = data[:,1] # 读取第二列数据
    tck = interpolate.splrep(x, y) # 进行样条插值
    xx = np.linspace(min(x), max(x), 1000)
    yy = interpolate.splev(xx, tck, der=0)
    plt.plot(x,y,'-',xx,yy,)
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.title(['file'])
    plt.savefig('{}.png'.format(file), dpi = 300)
    plt.close()
    count = count + 1


print("Total number of spectra: ", count)