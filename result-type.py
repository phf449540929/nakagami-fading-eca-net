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

x = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0]

'''
         0.0,       0.5,      1.0,     1.5,      2.0,      2.5,      3.0,      3.5,      4.0,     4.5,       5.0,      5.5,      6.0,      6.5,       7.0,     7.5,      8.0,      8.5,     9.0,       9.5,      10.0,     10.5,    11.0,     11.5,     12.0
'''

y1 = [0.966038, 0.983051, 0.984444, 0.985743, 0.988722, 0.990157, 0.990366, 0.991701, 0.991935, 0.992000, 0.992218, 0.993548, 0.994106, 0.994208, 0.994350, 0.994495, 0.995772, 0.995859, 0.995943, 0.995960, 0.996024, 0.996101, 0.997947, 0.997988, 0.998031]
y2 = [0.273063, 0.290909, 0.300626, 0.301370, 0.306418, 0.312860, 0.317287, 0.326172, 0.334623, 0.340681, 0.347422, 0.357553, 0.361789, 0.372934, 0.382277, 0.401942, 0.430085, 0.473610, 0.520256, 0.576923, 0.655882, 0.737850, 0.805882, 0.837850, 0.849964]
y3 = [0.230476, 0.242188, 0.247807, 0.253036, 0.257692, 0.268351, 0.281585, 0.287434, 0.294704, 0.297959, 0.299127, 0.303609, 0.304904, 0.307377, 0.316562, 0.328094, 0.367691, 0.438017, 0.543497, 0.667359, 0.773428, 0.827848, 0.851855, 0.867486, 0.871858]
y4 = [0.227969, 0.245098, 0.267974, 0.273293, 0.279528, 0.285592, 0.288172, 0.296218, 0.306613, 0.311741, 0.323278, 0.330059, 0.331313, 0.337972 ,0.367754, 0.432377, 0.548065, 0.647742, 0.709412, 0.737063, 0.759417, 0.775456, 0.793047, 0.802172, 0.807018]
y5 = [0.581116, 0.588762, 0.599213, 0.609378, 0.618268, 0.627385, 0.633891, 0.637993, 0.643132, 0.649425, 0.654971, 0.660348, 0.664717, 0.671078, 0.677937 ,0.688285, 0.702479, 0.713407, 0.732143, 0.752574, 0.775859, 0.798403, 0.816518, 0.824407, 0.833461]
y6 = [1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000]

# z1 = [0.5000, 0.6000, 0.70, 0.80, 0.90, 0.99, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# z2 = [0.2567, 0.2753, 0.32, 0.39, 0.57, 0.98, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# z3 = [0.2100, 0.2500, 0.29, 0.33, 0.42, 0.83, 0.98, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# z4 = [0.1200, 0.2300, 0.34, 0.45, 0.57, 0.68, 0.96, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

ax = plt.gca()

ax.plot(x, y1, marker='o', clip_on=False, label=u'bch', mfc=(0.1217, 0.4667, 0.7059, 0),
        mec=(0.1217, 0.4667, 0.7059))
ax.plot(x, y2, marker='s', clip_on=False, label=u'rs', mfc=(1.0000, 0.4980, 0.0549, 0),
        mec=(1.0000, 0.4980, 0.0549))
ax.plot(x, y3, marker='v', clip_on=False, label=u'conv', mfc=(0.1725, 0.6275, 0.1725, 0),
        mec=(0.1725, 0.6275, 0.1725))
ax.plot(x, y4, marker='x', clip_on=False, label=u'ldpc', mfc=(0.8392, 0.1529, 0.1569, 0),
        mec=(0.8392, 0.1529, 0.1569))
ax.plot(x, y5, marker='+', clip_on=False, label=u'turbo', mfc=(0.5804, 0.4039, 0.7411, 0),
        mec=(0.5804, 0.4039, 0.7411))
ax.plot(x, y6, marker='^', clip_on=False, label=u'polar', mfc=(0.5451, 0.2706, 0.0745, 0),
        mec=(0.5451, 0.2706, 0.0745))

# ax.plot(x, z1, marker='o', clip_on=False, label=u'L1 Ref [16]', mfc=(0, 0, 0, 0), mec=(0, 0, 0), color='k',
#         linestyle='dashed')
# ax.plot(x, z2, marker='s', clip_on=False, label=u'L2 Ref [16]', mfc=(0, 0, 0, 0), mec=(0, 0, 0), color='k',
#         linestyle='dashed')
# ax.plot(x, z3, marker='v', clip_on=False, label=u'L3 Ref [16]', mfc=(0, 0, 0, 0), mec=(0, 0, 0), color='k',
#         linestyle='dashed')
# ax.plot(x, z4, marker='x', clip_on=False, label=u'L4 Ref [16]', mfc=(0, 0, 0, 0), mec=(0, 0, 0), color='k',
#         linestyle='dashed')

ax.legend(loc='lower right')  # 让图例生效

ax.set_xlim(0, 12)
ax.set_ylim(0, 1)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

ax.set_xlabel(u"SNR(dB)")
ax.set_ylabel(u"Probability of Detection")

plt.margins(0)

plt.show()
