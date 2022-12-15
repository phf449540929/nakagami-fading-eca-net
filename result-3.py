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
         0          2         4         6        8        10         12        14       16        18        20     22      24       26      28     30
'''
y1 = [0.260714, 0.296000, 0.342540, 0.401013, 0.469247, 0.549729, 0.622568, 0.685738, 0.736000, 0.766860, 0.7967, 0.8095, 0.8230, 0.8339, 0.8449, 0.8491]
y2 = [0.071860, 0.108333, 0.153734, 0.208095, 0.278125, 0.365397, 0.473744, 0.555783, 0.628233, 0.699248, 0.7371, 0.7633, 0.7850, 0.7986, 0.8067, 0.8148]
y3 = [0.271654, 0.279738, 0.297333, 0.329182, 0.361487, 0.397562, 0.459459, 0.555556, 0.638298, 0.715447, 0.7533, 0.7777, 0.7981, 0.8095, 0.8236, 0.8291]
y4 = [0.231076, 0.255226, 0.283173, 0.315124, 0.348814, 0.395738, 0.457160, 0.522042, 0.603276, 0.678862, 0.7411, 0.7810, 0.8107, 0.8243, 0.8368, 0.8430]

'''
        0     2     4     6     8    10    12    14    16    18     20    22    24    26   28    30
'''
z1 = [0.11, 0.14, 0.19, 0.27, 0.36, 0.46, 0.54, 0.61, 0.67, 0.71, 0.74, 0.76, 0.78, 0.79, 0.80, 0.80]
z2 = [0.05, 0.07, 0.10, 0.14, 0.22, 0.33, 0.43, 0.54, 0.61, 0.66, 0.70, 0.73, 0.75, 0.77, 0.78, 0.79]
z3 = [0.09, 0.11, 0.15, 0.21, 0.29, 0.37, 0.45, 0.54, 0.60, 0.64, 0.68, 0.71, 0.74, 0.76, 0.78, 0.79]
z4 = [0.07, 0.08, 0.10, 0.14, 0.21, 0.30, 0.41, 0.51, 0.60, 0.67, 0.72, 0.75, 0.77, 0.79, 0.80, 0.81]

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
