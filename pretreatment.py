#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    @File:         pretreatment_llr_cal.py
    @Author:       haifeng
    @Since:        python3.9
    @Version:      V1.0
    @Date:         2021/9/13 21:38
    @Description:
-------------------------------------------------
    @Change:
        2021/9/13 21:38
-------------------------------------------------
"""

import math
import sys

import pandas
import numpy
from sklearn import model_selection, preprocessing
from sklearn.preprocessing import MinMaxScaler


def get_array(file_name):
    dataframe = pandas.read_csv(file_name)

    text_array = pandas.DataFrame(dataframe['text'])['text']

    llr_array = []
    for i in range(0, len(text_array)):
        s = text_array[i]
        s = s.strip()
        s = s[0:-1]
        llr_array.append(s.split('/'))

    max_len = 0
    for i in range(0, len(llr_array)):
        if len(llr_array[i]) > max_len:
            max_len = len(llr_array[i])
    print("max_len = " + str(max_len))

    return dataframe, llr_array, max_len


def write_in_csv(x, y, train_or_validate, file_name):
    dataframe_write = pandas.DataFrame()
    x_str = []
    for row in x:
        string = ""
        for a in row:
            string += str(a)
            string += " "
        x_str.append(string)
    dataframe_write['X'] = x_str
    dataframe_write['Y'] = y

    index = file_name[:-4]

    if train_or_validate == 1:
        dataframe_write.to_csv(index + '-pre-raw-train.csv', index=True, sep=',')
    else:
        dataframe_write.to_csv(index + '-pre-raw-validate.csv', index=True, sep=',')


def do_pretreatment(file_name):
    dataframe, llr_array, max_len = get_array(file_name)

    for i in range(0, len(llr_array)):
        a = llr_array[i]
        k = len(a)
        while k < 4096:
            a.append('0')
            k += 1
        if k > 4096:
            a = a[0:4096]
        llr_array[i] = a
    print("the length of llr_array[0]: " + str(len(llr_array[0])))

    for i in range(len(llr_array)):
        # print(len(llr_array[0]))
        for j in range(len(llr_array[i])):
            if llr_array[i][j] == 'Inf':
                llr_array[i][j] = 999.0
            if llr_array[i][j] == '-Inf':
                llr_array[i][j] = -999.0
            llr_array[i][j] = float(llr_array[i][j])

    scale = MinMaxScaler()
    result_matrix = scale.fit_transform(llr_array)

    x = result_matrix
    y = dataframe['label']

    x_train, x_valid, y_train, y_valid = model_selection.train_test_split(x, y)

    write_in_csv(x, y, 1, file_name)
    write_in_csv(x_valid, y_valid, 2, file_name)


def main():
    # snr = sys.argv[1]
    # coding = 'conv'
    # coding = 'ldpc'
    # coding = 'polar'
    # coding = 'turbo'
    # file_name = "dataset-awgn-ldpc--10db.csv"
    # file_name = "dataset-awgn-" + coding + "-" + str(snr) + "db.csv"
    # file_name = "dataset-awgn-conv--10db.csv"

    for i in range(-10, 22, 2):
        file_name = "./dataset/rayleigh/dataset-rayleigh-ldpc-" + str(i) + "db.csv"
        do_pretreatment(file_name)


if __name__ == '__main__':
    main()
