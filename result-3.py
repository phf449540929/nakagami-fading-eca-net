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
         0      2       4       6        8       10     12       14     16      18       20     22      24       26      28     30
'''

y1 = [0.4419, 0.4632, 0.5075, 0.5750, 0.6477, 0.7118, 0.7556, 0.7943, 0.8182, 0.8430, 0.8567, 0.8695, 0.8830, 0.8939, 0.9049, 0.9091]
y2 = [0.1605, 0.1841, 0.2280, 0.3006, 0.3719, 0.4410, 0.5074, 0.5979, 0.6868, 0.7600, 0.8071, 0.8333, 0.8750, 0.8986, 0.9167, 0.9248]
y3 = [0.3400, 0.3617, 0.3877, 0.4203, 0.4737, 0.5300, 0.6085, 0.6765, 0.7368, 0.7767, 0.8133, 0.8437, 0.8681, 0.8875, 0.8936, 0.9091]
y4 = [0.2929, 0.3315, 0.3852, 0.4619, 0.5447, 0.6133, 0.6740, 0.7264, 0.7623, 0.7939, 0.8311, 0.8510, 0.8757, 0.8893, 0.9018, 0.9130]

z1 = [0.49, 0.50, 0.53, 0.57, 0.61, 0.66, 0.70, 0.74, 0.77, 0.80, 0.82, 0.840, 0.85, 0.86, 0.87, 0.87]
z2 = [0.14, 0.15, 0.17, 0.26, 0.36, 0.46, 0.55, 0.62, 0.69, 0.74, 0.79, 0.820, 0.84, 0.86, 0.87, 0.88]
z3 = [0.19, 0.20, 0.22, 0.28, 0.36, 0.44, 0.52, 0.59, 0.65, 0.71, 0.75, 0.785, 0.81, 0.83, 0.85, 0.86]
z4 = [0.43, 0.44, 0.46, 0.49, 0.52, 0.55, 0.59, 0.63, 0.68, 0.75, 0.78, 0.820, 0.84, 0.86, 0.87, 0.88]

ax = plt.gca()

ax.plot(x, y1, marker='o', clip_on=False, label=u'L1', mfc=(0.1217, 0.4667, 0.7059, 0),
        mec=(0.1217, 0.4667, 0.7059))
ax.plot(x, y2, marker='s', clip_on=False, label=u'L2', mfc=(1.0000, 0.4980, 0.0549, 0),
        mec=(1.0000, 0.4980, 0.0549))
ax.plot(x, y3, marker='v', clip_on=False, label=u'L3', mfc=(0.1725, 0.6275, 0.1725, 0),
        mec=(0.1725, 0.6275, 0.1725))
ax.plot(x, y4, marker='x', clip_on=False, label=u'L4', mfc=(0.8392, 0.1529, 0.1569, 0),
        mec=(0.8392, 0.1529, 0.1569))

ax.plot(x, z1, marker='o', clip_on=False, label=u'L1 Ref[37]', mfc=(0, 0, 0, 0), mec=(0, 0, 0), color='k',
        linestyle='dashed')
ax.plot(x, z2, marker='s', clip_on=False, label=u'L2 Ref[37]', mfc=(0, 0, 0, 0), mec=(0, 0, 0), color='k',
        linestyle='dashed')
ax.plot(x, z3, marker='v', clip_on=False, label=u'L3 Ref[37]', mfc=(0, 0, 0, 0), mec=(0, 0, 0), color='k',
        linestyle='dashed')
ax.plot(x, z4, marker='x', clip_on=False, label=u'L4 Ref[37]', mfc=(0, 0, 0, 0), mec=(0, 0, 0), color='k',
        linestyle='dashed')

ax.legend(loc='lower right')  # 让图例生效

ax.set_xlim(0, 30)
ax.set_ylim(0, 1)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

ax.set_xlabel(u"SNR(dB)")
ax.set_ylabel(u"Probability of Detection")

plt.margins(0)

plt.show()
