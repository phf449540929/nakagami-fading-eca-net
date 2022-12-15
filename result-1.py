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

x = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

'''
         -10      -8       -6          -4       -2        0      2    4    6    8   10   12   14   16   18   20
'''

y1 = [0.466265, 0.510781, 0.576132, 0.727642, 0.971074, 1.0000, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
y2 = [0.290798, 0.355102, 0.476744, 0.628352, 0.804598, 1.0000, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
y3 = [0.354701, 0.385171, 0.427015, 0.510816, 0.708502, 1.0000, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
y4 = [0.389764, 0.418983, 0.488139, 0.596774, 0.808000, 1.0000, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

z1 = [0.5000, 0.6000, 0.70, 0.80, 0.90, 0.99, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
z2 = [0.2567, 0.2753, 0.32, 0.39, 0.57, 0.98, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
z3 = [0.2100, 0.2500, 0.29, 0.33, 0.42, 0.83, 0.98, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
z4 = [0.1200, 0.2300, 0.34, 0.45, 0.57, 0.68, 0.96, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

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

ax.set_xlim(-10, 20)
ax.set_ylim(0, 1)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

ax.set_xlabel(u"SNR(dB)")
ax.set_ylabel(u"Probability of Detection")

plt.margins(0)

plt.show()
