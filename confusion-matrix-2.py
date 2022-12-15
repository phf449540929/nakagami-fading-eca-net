#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    @File:         confusion-matrix
    @Author:       haifeng
    @Since:        python3.9
    @Version:      V1.0
    @Date:         2022/2/22 20:57
    @Description:  
-------------------------------------------------
    @Change:
        2022/2/22 20:57
-------------------------------------------------
"""

import matplotlib.pyplot as plt
import numpy as np

confusion = np.array(([0.6715, 0.0977, 0.0357, 0.1951],
                      [0.0499, 0.7603, 0.1259, 0.0639],
                      [0.0894, 0.0212, 0.7703, 0.1191],
                      [0.1892, 0.1208, 0.0681, 0.6219]))

for i in range(len(confusion)):
    print(str(confusion[i][0]) + " + " + str(confusion[i][1]) + " + " + str(confusion[i][2]) + " + " + str(
        confusion[i][3]) + " = " + str(confusion[i][0] + confusion[i][1] + confusion[i][2] + confusion[i][3]))
for j in range(len(confusion)):
    print(str(confusion[0][j]) + " + " + str(confusion[1][j]) + " + " + str(confusion[2][j]) + " + " + str(
        confusion[3][j]) + " = " + str(confusion[0][j] + confusion[1][j] + confusion[2][j] + confusion[3][j]))

# 热度图，后面是指定的颜色块，可设置其他的不同颜色
plt.imshow(confusion, cmap=plt.cm.Blues)
# ticks 坐标轴的坐标点
# label 坐标轴标签说明
indices = range(len(confusion))
# 第一个是迭代对象，表示坐标的显示顺序，第二个参数是坐标轴显示列表
# plt.xticks(indices, [0, 1, 2])
# plt.yticks(indices, [0, 1, 2])
plt.xticks(indices, ['L1', 'L2', 'L3', 'L4'])
plt.yticks(indices, ['L1', 'L2', 'L3', 'L4'])

plt.colorbar()

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
# plt.title('(1)Confusion Matrix AWGN')

# plt.rcParams两行是用于解决标签不能显示汉字的问题
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

thresh = confusion.max() / 2.

# 显示数据
for first_index in range(len(confusion)):  # 第几行
    for second_index in range(len(confusion[first_index])):  # 第几列
        plt.text(second_index, first_index, confusion[first_index][second_index], horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if confusion[first_index][second_index] > thresh else "black")
# 在matlab里面可以对矩阵直接imagesc(confusion)
# 显示
plt.show()
