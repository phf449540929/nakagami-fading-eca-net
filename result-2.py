#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    @File:         result-awgn-conv
    @Author:       haifeng
    @Since:        python3.9
    @Version:      V1.0
    @Date:         2021/10/17 16:07
    @Description:  
-------------------------------------------------
    @Change:
        2021/10/17 16:07
-------------------------------------------------
"""

# encoding=utf-8
import matplotlib.pyplot as plt
import numpy
from matplotlib import ticker

x = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]

'''
         0          2        4         6        8          10        12       14      16      18      20      22      24      26      28      30  
'''

y1 = [0.429655, 0.467148, 0.521805, 0.574899, 0.629242, 0.671538, 0.717869, 0.7638, 0.7993, 0.8283, 0.8566, 0.8755, 0.8888, 0.8933, 0.9043, 0.9071]
y2 = [0.238462, 0.300312, 0.372882, 0.473710, 0.620508, 0.760303, 0.830000, 0.8677, 0.8925, 0.9059, 0.9163, 0.9254, 0.9368, 0.9455, 0.9516, 0.9556]
y3 = [0.451295, 0.506024, 0.567347, 0.633846, 0.709512, 0.770213, 0.812063, 0.8482, 0.8688, 0.8894, 0.9093, 0.9171, 0.9237, 0.9283, 0.9333, 0.9373]
y4 = [0.391304, 0.405000, 0.433414, 0.470204, 0.531890, 0.621890, 0.746888, 0.8322, 0.8792, 0.9071, 0.9239, 0.9362, 0.9479, 0.9589, 0.9691, 0.9731]

z1 = [0.37, 0.39, 0.42, 0.46, 0.52, 0.58, 0.65, 0.70, 0.74, 0.77, 0.79, 0.81, 0.82, 0.83, 0.84, 0.84]
z2 = [0.04, 0.08, 0.14, 0.24, 0.35, 0.44, 0.52, 0.61, 0.68, 0.73, 0.77, 0.80, 0.83, 0.84, 0.85, 0.85]
z3 = [0.54, 0.52, 0.52, 0.54, 0.57, 0.61, 0.65, 0.71, 0.76, 0.80, 0.83, 0.86, 0.88, 0.90, 0.91, 0.92]
z4 = [0.33, 0.35, 0.37, 0.40, 0.43, 0.47, 0.51, 0.57, 0.64, 0.70, 0.75, 0.79, 0.82, 0.84, 0.85, 0.86]

ax = plt.gca()

ax.plot(x, y1, marker='o', clip_on=False, label=u'L1', mfc=(0.1217, 0.4667, 0.7059, 0),
        mec=(0.1217, 0.4667, 0.7059))
ax.plot(x, y2, marker='s', clip_on=False, label=u'L2', mfc=(1.0000, 0.4980, 0.0549, 0),
        mec=(1.0000, 0.4980, 0.0549))
ax.plot(x, y3, marker='v', clip_on=False, label=u'L3', mfc=(0.1725, 0.6275, 0.1725, 0),
        mec=(0.1725, 0.6275, 0.1725))
ax.plot(x, y4, marker='x', clip_on=False, label=u'L4', mfc=(0.8392, 0.1529, 0.1569, 0),
        mec=(0.8392, 0.1529, 0.1569))

ax.plot(x, z1, marker='o', clip_on=False, label=u'L1 Ref [16]', mfc=(0, 0, 0, 0), mec=(0, 0, 0), color='k',
        linestyle='dashed')
ax.plot(x, z2, marker='s', clip_on=False, label=u'L2 Ref [16]', mfc=(0, 0, 0, 0), mec=(0, 0, 0), color='k',
        linestyle='dashed')
ax.plot(x, z3, marker='v', clip_on=False, label=u'L3 Ref [16]', mfc=(0, 0, 0, 0), mec=(0, 0, 0), color='k',
        linestyle='dashed')
ax.plot(x, z4, marker='x', clip_on=False, label=u'L4 Ref [16]', mfc=(0, 0, 0, 0), mec=(0, 0, 0), color='k',
        linestyle='dashed')

ax.legend(loc='lower right')  # 让图例生效

ax.set_xlim(0, 30)
ax.set_ylim(0, 1)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

ax.set_xlabel(u"SNR(dB)")
ax.set_ylabel(u"Probability of Detection")

plt.margins(0)

plt.show()
