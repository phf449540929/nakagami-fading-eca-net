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

y1 = [0.3512, 0.4088, 0.4586, 0.5306, 0.5957, 0.6609, 0.7148, 0.7538, 0.7893, 0.8183, 0.8366, 0.8455, 0.8488, 0.8533, 0.8543, 0.8571]
y2 = [0.2080, 0.2667, 0.3458, 0.4143, 0.4868, 0.5668, 0.6613, 0.7377, 0.8025, 0.8529, 0.8813, 0.9054, 0.9168, 0.9255, 0.9316, 0.9356]
y3 = [0.3880, 0.4189, 0.4538, 0.5143, 0.5768, 0.6500, 0.7173, 0.7582, 0.7888, 0.8194, 0.8493, 0.8671, 0.8837, 0.8933, 0.8933, 0.8933]
y4 = [0.3402, 0.3907, 0.4489, 0.5083, 0.5763, 0.6328, 0.6918, 0.7422, 0.7792, 0.8171, 0.8389, 0.8562, 0.8809, 0.8989, 0.9091, 0.9091]

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
