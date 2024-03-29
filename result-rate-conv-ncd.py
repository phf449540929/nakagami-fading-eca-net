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
y1 = [0.003469, 0.003915, 0.019577, 0.055825, 0.109023, 0.197582, 0.375047, 0.556115, 0.671974, 0.742245, 0.770731, 0.786781, 0.794696, 0.796500, 0.798289, 0.800376, 0.802000, 0.804414, 0.806311, 0.808039, 0.810206, 0.812288, 0.814469, 0.816993, 0.818042]
y2 = [0.006347, 0.009667, 0.006087, 0.006713, 0.007562, 0.043439, 0.109155, 0.219744, 0.386131, 0.561041, 0.675301, 0.740195, 0.773393, 0.785697, 0.787804, 0.789950, 0.791946, 0.793552, 0.795297, 0.797224, 0.799928, 0.801217, 0.803136, 0.805701, 0.807722]
y3 = [0.002022, 0.000801, 0.001444, 0.000907, 0.000248, 0.005887, 0.246760, 0.457803, 0.605064, 0.689507, 0.731053, 0.750375, 0.760755, 0.762446, 0.764558, 0.766273, 0.768484, 0.770288, 0.772804, 0.774097, 0.776209, 0.778232, 0.780370, 0.782497, 0.784716]
y4 = [0.006253, 0.009688, 0.009433, 0.005676, 0.002273, 0.000370, 0.201571, 0.390240, 0.534802, 0.626981, 0.678725, 0.715100, 0.736267, 0.744533, 0.751179, 0.753013, 0.755760, 0.757456, 0.759018, 0.761424, 0.763861, 0.765950, 0.767171, 0.769338, 0.771146]

# y1 = [0.434783, 0.597765, 0.682136, 0.734921, 0.763908, 0.777480, 0.786471, 0.792879, 0.798935, 0.803030, 0.806701, 0.806004, 0.806648, 0.806747, 0.806597, 0.806879, 0.806854, 0.806308, 0.806455, 0.806861, 0.806324, 0.806977, 0.806279, 0.806463, 0.806441]
# y2 = [0.493113, 0.571461, 0.629053, 0.656809, 0.680077, 0.693871, 0.705984, 0.706008, 0.709223, 0.709412, 0.709221, 0.709182, 0.709990, 0.709257, 0.709170, 0.709979, 0.709502, 0.709019, 0.709702, 0.709726, 0.709986, 0.709169, 0.709146, 0.709111, 0.709105]
# y3 = [0.398230, 0.580603, 0.652686, 0.699794, 0.725974, 0.742606, 0.751186, 0.758667, 0.762821, 0.763908, 0.764380, 0.764817, 0.764613, 0.764613, 0.764105, 0.764270, 0.764471, 0.764039, 0.764980, 0.764730, 0.764250, 0.764122, 0.764463, 0.764132, 0.764655]
# y4 = [0.490040, 0.592371, 0.664872, 0.698577, 0.719205, 0.742997, 0.757381, 0.769494, 0.773333, 0.779826, 0.780671, 0.781462, 0.781756, 0.781701, 0.781734, 0.781804, 0.781569, 0.781179, 0.781990, 0.781255, 0.781585, 0.781500, 0.781645, 0.781238, 0.781231]

# z1 = [0.5000, 0.6000, 0.70, 0.80, 0.90, 0.99, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# z2 = [0.2567, 0.2753, 0.32, 0.39, 0.57, 0.98, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# z3 = [0.2100, 0.2500, 0.29, 0.33, 0.42, 0.83, 0.98, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# z4 = [0.1200, 0.2300, 0.34, 0.45, 0.57, 0.68, 0.96, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

ax = plt.gca()

ax.plot(x, y1, marker='o', clip_on=False, label=u'rate = 1/2 Ref [24]', mfc=(0.1217, 0.4667, 0.7059, 0),
        mec=(0.1217, 0.4667, 0.7059))
ax.plot(x, y2, marker='s', clip_on=False, label=u'rate = 2/3 Ref [24]', mfc=(1.0000, 0.4980, 0.0549, 0),
        mec=(1.0000, 0.4980, 0.0549))
ax.plot(x, y3, marker='v', clip_on=False, label=u'rate = 3/4 Ref [24]', mfc=(0.1725, 0.6275, 0.1725, 0),
        mec=(0.1725, 0.6275, 0.1725))
ax.plot(x, y4, marker='x', clip_on=False, label=u'rate = 5/6 Ref [24]', mfc=(0.8392, 0.1529, 0.1569, 0),
        mec=(0.8392, 0.1529, 0.1569))
# ax.plot(x, y5, marker='+', clip_on=False, label=u'length = 1944', mfc=(0.5804, 0.4039, 0.7411, 0),
#         mec=(0.5804, 0.4039, 0.7411))
# ax.plot(x, y6, marker='^', clip_on=False, label=u'length = 2304', mfc=(0.5451, 0.2706, 0.0745, 0),
#         mec=(0.5451, 0.2706, 0.0745))

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
