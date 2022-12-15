#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    @File:         proprocess
    @Author:       haifeng
    @Since:        python3.9
    @Version:      V1.0
    @Date:         2021/7/6 21:27
    @Description:  
-------------------------------------------------
    @Change:
        2021/7/6 21:27
-------------------------------------------------
"""
import math

import pandas
import numpy
from sklearn import model_selection, preprocessing
from sklearn.preprocessing import MinMaxScaler


# def do_pretreatment(file_name):
#     dataframe, llr_array, max_len = get_array(file_name)
#
#     for i in range(0, len(llr_array)):
#         a = llr_array[i]
#         k = len(a)
#         while k < 4096:
#             a.append('0')
#             k += 1
#         if k > 4096:
#             a = a[0:4096]
#         llr_array[i] = a
#     print("the length of llr_array[0]: " + str(len(llr_array[0])))
#
#     for i in range(len(llr_array)):
#         # print(len(llr_array[0]))
#         for j in range(len(llr_array[i])):
#             if llr_array[i][j] == 'Inf':
#                 llr_array[i][j] = 999.0
#             if llr_array[i][j] == '-Inf':
#                 llr_array[i][j] = -999.0
#             llr_array[i][j] = float(llr_array[i][j])
#
#     scale = MinMaxScaler()
#     result_matrix = scale.fit_transform(llr_array)
#
#     x = result_matrix
#     y = dataframe['label']
#
#     x_train, x_valid, y_train, y_valid = model_selection.train_test_split(x, y)
#
#     write_in_csv(x, y, 1, file_name)
#     write_in_csv(x_valid, y_valid, 2, file_name)


def feature_extraction(file_name):
    dataframe = pandas.read_csv(file_name)

    text_array = pandas.DataFrame(dataframe['text'])['text']
    '''
    n * 16385
    '''

    max_len = 16385

    GF_p = 2
    GF_n = 1
    while (GF_p ** GF_n - 1) * GF_n < max_len:
        GF_n += 1
    GF_n -= 1
    row_number = GF_p ** GF_n - 1
    line_length = row_number * GF_n
    print("GF_p = " + str(GF_p))
    print("GF_n = " + str(GF_n))
    print("GF_p ^ GF_n - 1 = " + str(row_number))
    print("line_length = " + str(line_length))
    '''
    1023 * 10
    '''

    for i in range(0, len(text_array)):
        text_array[i] = text_array[i].strip()
        text_array[i] = text_array[i][:line_length]
    print("the length of text_array[0]: " + str(len(text_array[0])))
    '''
    n * 10230
    '''

    '''
    runs
    --------------------------------------------------------------------------------------------------------------------
    '''
    runs_matrix = []
    for line in text_array:
        i = int(line_length / row_number)
        col = []
        for j in range(0, row_number):
            row = line[j * i:(j + 1) * i]

            runs = 0
            for k in range(0, len(row) - 1):
                if row[k] != row[k + 1]:
                    runs += 1
            runs += 1
            col.append(runs)
        runs_matrix.append(col)

    print("the length of the runs_matrix: " + str(len(runs_matrix)))
    print("the length of a col vector in the runs_matrix: " + str(len(runs_matrix[0])))
    '''
    n * 1023
    '''

    '''
    the depth of the spectrum
    --------------------------------------------------------------------------------------------------------------------
    '''
    depth_matrix = []
    for line in text_array:
        i = int(line_length / row_number)
        col = []
        for j in range(0, row_number):
            row = line[j * i:(j + 1) * i]

            a = row
            depth = 0
            while 1:
                flag = 0
                for k in a:
                    if int(k) != 0:
                        flag = 1
                if not flag:
                    break

                b = []
                for k in range(0, len(a) - 1):
                    b.append(int(a[k]) ^ int(a[k + 1]))
                a = b
                depth += 1

            col.append(depth)
        depth_matrix.append(col)

    print("the length of the depth_matrix: " + str(len(depth_matrix)))
    print("the length of a col vector in the depth_matrix: " + str(len(depth_matrix[0])))
    '''
    n * 1023
    '''

    '''
    linear complexity
    --------------------------------------------------------------------------------------------------------------------
    '''
    linear_matrix = []
    miu = 10 / 2 + (9 + (-1) ** (10 + 1)) / 36 - (10 / 3 + 2 / 9) / 2 ** 10
    for line in text_array:
        x = int(line_length / row_number)
        linear_list = []
        for y in range(0, row_number):
            row = line[y * x:(y + 1) * x]

            q = 2
            # a = [0, 0, 1, 0, 1, 1, 0, 1]
            a = []
            for k in range(0, len(row)):
                a.append(int(row[k]))

            d = []
            f = []
            l = []
            for i in range(len(a) + 1):
                d.append(-100000)
                f.append([])
                l.append(-100000)

            f[0] = [1]
            l[0] = 0
            for n0 in range(len(a)):
                if a[n0] != 0:
                    break

            for i in range(n0):
                d[i] = 0
            d[n0] = a[n0]

            for i in range(1, n0 + 1):
                f[i] = [1]
                l[i] = 0

            temp_f = [1]
            for i in range(1, n0 + 1):
                temp_f.append(0)
            temp_f.append(0 - d[n0])
            f[n0 + 1] = temp_f
            l[n0 + 1] = n0 + 1

            for n in range(n0 + 1, len(a)):
                d[n] = 0
                for i in range(l[n] + 1):
                    d[n] += a[n - i] * f[n][i]
                d[n] = d[n] % q
                if d[n] == 0:
                    f[n + 1] = f[n]
                    l[n + 1] = l[n]
                else:
                    for m in range(n - 1, -1, -1):
                        if l[m] < l[m + 1]:
                            break

                    l[n + 1] = max(l[n], n + 1 - l[n])

                    temp_f = []
                    temp_fn = f[n].copy()
                    temp_fm = f[m].copy()
                    for i in range(l[n + 1] + 1):
                        if i > l[n]:
                            temp_fn.append(0)
                        if i > l[m]:
                            temp_fm.append(0)
                    k = n - m
                    pre = []
                    for i in range(k):
                        pre.append(0)
                    pre.extend(temp_fm[0:l[n + 1] + 1 - k])
                    temp_fm = pre
                    f[n + 1] = []

                    for i in range(l[n + 1] + 1):
                        f[n + 1].append(temp_fn[i] - temp_fm[i])

            # print('a:', a)
            # print('f(x):', f)
            # print('l:', l)
            # print('d:', d)
            # print(l[-1])

            linear_list.append((-1) ** 10 * (l[-1] - miu))
            # linear_list.append((-1) ** 10 * (l[-1] - miu) + 2 / 9)
        linear_matrix.append(linear_list)

    print("the length of the linear_matrix: " + str(len(linear_matrix)))
    print("the length of a col vector in the linear_matrix: " + str(len(linear_matrix[0])))
    '''
    n * 1023
    '''

    '''
    code weight similarity
    --------------------------------------------------------------------------------------------------------------------
    '''
    random_weight_list = []
    for i in range(0, 11):
        up = 1
        low = 1
        m = 10
        n = 1
        for j in range(0, i):
            up *= m
            low *= n
            m -= 1
            n += 1
        random_weight_list.append(up / low / 1024)

    d2 = 0
    for i in range(0, 11):
        d2 += (random_weight_list[i] - 1 / 11) ** 2
    d2 /= 11

    weight_matrix = []
    for line in text_array:
        i = int(line_length / row_number)
        weight_probability_list = []
        for j in range(0, row_number):
            row = line[j * i:(j + 1) * i]
            weight = 0
            for k in row:
                if k == '1':
                    weight += 1
            weight_list = [0 for v in range(11)]
            weight_list[weight] = 1

            d1 = 0
            for k in range(0, 11):
                d1 += (weight_list[k] - 1 / 11) ** 2
            d1 /= 11

            cov = 0
            for k in range(0, 11):
                cov += (weight_list[k] - 1 / 11) * (random_weight_list[k] - 1 / 11)

            weight_probability_list.append((cov / (d1 * d2) ** 0.5) ** 2)
        weight_matrix.append(weight_probability_list)

    print("the length of the weight_matrix: " + str(len(weight_matrix)))
    print("the length of a col vector in the weight_matrix: " + str(len(weight_matrix[0])))
    '''
    n * 1023
    '''

    '''
    sequence autocorrelation
    --------------------------------------------------------------------------------------------------------------------
    '''
    correlation_matrix = []
    for line in text_array:
        i = int(line_length / row_number)
        col = []
        for j in range(0, row_number):
            row = line[j * i:(j + 1) * i]
            correlation = 0
            for k in range(0, len(row) - 2):
                correlation += int(row[k]) ^ int(row[k + 2])
            col.append(correlation)
        correlation_matrix.append(col)

    print("the length of the correlation_matrix: " + str(len(correlation_matrix)))
    print("the length of a col vector in the correlation_matrix: " + str(len(correlation_matrix[0])))
    '''
    n * 1023
    '''

    '''
    discrete fourier transformation
    --------------------------------------------------------------------------------------------------------------------
    '''
    # sample dots' number
    N = row_number
    # the source signal
    # contain the basis exp(-j*2*pi/N*k*n) and the projection weight
    fourier_matrix = []
    for k in range(N):
        basis = []
        for n in range(N):
            basis.append(complex(math.cos(2 * math.pi / N * k * n), math.sin(2 * math.pi / N * k * n)))
        fourier_matrix.append(basis)

    print("the length of the fourier_matrix: " + str(len(fourier_matrix)))
    print("the length of a col vector in the fourier_matrix: " + str(len(fourier_matrix[0])))
    '''
    1023 * 1023
    '''

    convert_matrix = []
    for line in text_array:
        i = int(line_length / row_number)
        col = []
        for j in range(0, row_number):
            row = line[j * i:(j + 1) * i]
            a = int(row, 2)
            col.append(a)
        convert_matrix.append(numpy.dot(fourier_matrix, numpy.array([col]).T))

    print("the length of the convert_matrix: " + str(len(convert_matrix)))
    print("the length of a col vector in the convert_matrix: " + str(len(convert_matrix[0])))
    '''
    n * 1023
    '''

    fourier_feature_real_matrix = []
    fourier_feature_imag_matrix = []
    for line in convert_matrix:
        real_col = []
        imag_col = []
        for i in line:
            real_col.append(i.real[0])
            imag_col.append(i.imag[0])
        fourier_feature_real_matrix.append(real_col)
        fourier_feature_imag_matrix.append(imag_col)

    print("the length of the fourier_feature_real_matrix: " + str(len(fourier_feature_real_matrix)))
    print("the length of a col vector in the fourier_feature_real_matrix: " + str(len(fourier_feature_real_matrix[0])))
    print("the length of the fourier_feature_imag_matrix: " + str(len(fourier_feature_imag_matrix)))
    print("the length of a col vector in the fourier_feature_imag_matrix: " + str(len(fourier_feature_imag_matrix[0])))
    '''
    n * 2 * 1023
    '''

    # scale = MinMaxScaler()
    # runs_matrix = scale.fit_transform(runs_matrix)
    # depth_matrix = scale.fit_transform(depth_matrix)
    # linear_matrix = scale.fit_transform(linear_matrix)
    # weight_matrix = scale.fit_transform(weight_matrix)
    # correlation_matrix = scale.fit_transform(correlation_matrix)
    # fourier_feature_real_matrix = scale.fit_transform(fourier_feature_real_matrix)
    # fourier_feature_imag_matrix = scale.fit_transform(fourier_feature_imag_matrix)

    feature_matrix = []
    for i in range(0, len(text_array)):
        feature = [runs_matrix[i], depth_matrix[i], linear_matrix[i], weight_matrix[i], correlation_matrix[i],
                   fourier_feature_real_matrix[i], fourier_feature_imag_matrix[i]]
        feature_matrix.append(feature)


    feature_matrix = numpy.array(feature_matrix).reshape(len(feature_matrix), 7 * 1023)

    scale = MinMaxScaler()
    feature_matrix = scale.fit_transform(feature_matrix)

    # feature_matrix = numpy.array(feature_matrix).reshape(len(feature_matrix), 7, 1, 1023)

    x = feature_matrix
    y = dataframe['label']

    return x, y

def write_in_csv(x, y, train_or_validate, file):

    dataframe_write = pandas.DataFrame()
    x_str = []
    for row in x:
        string = ""
        for a in row:
            string += str(a)
            string += " "
        x_str.append(string)
    dataframe_write['X'] = x_str
    dataframe_write['Y'] = numpy.array(y)

    index = file[:-4]

    if train_or_validate == 1:
        dataframe_write.to_csv(index + '-pre-train.csv', index=True, sep=',')
    else:
        dataframe_write.to_csv(index + '-pre-validate.csv', index=True, sep=',')


if __name__ == '__main__':
    # 3.0, 3.5, 4.0,
    snr = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5,
           4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5,
           8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5,
           12.0]
    for g in range(0, len(snr)):
        file_name = "./dataset/rate/conv/dataset-rate-conv-" + str(snr[g]) + "db.csv"
        res1, res2 = feature_extraction(file_name)
        x_train, x_valid, y_train, y_valid = model_selection.train_test_split(res1, res2)
        write_in_csv(x_train, y_train, 1, file_name)
        write_in_csv(x_valid, y_valid, 2, file_name)

    # res1, res2, res3, res4 = feature_extraction(file_name)
    #
    # print(len(res1))
    # print(res1[0])
    # print(len(res1[0]))
    # print(len(res1[0][0]))
    #
    # print(len(res2))
    # print(len(res2[0]))
    # print(len(res2[0][0]))
    #
    # print(len(res3))
    #
    # print(len(res4))
