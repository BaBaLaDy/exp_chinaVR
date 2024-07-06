#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Oct  8 16:05:33 2019

@author: chengshuoxia
"""
import glob
import itertools
import csv
import pandas as pd
import os
from scipy import stats
import numpy as np


#########################################################################

def seg(data_path, name, new_folder, sl, ID):
    orin_name = "*" + name + "*"
    # inputfile = glob.glob(data_path + '/' + orin_name + ".csv")
    inputfile = data_path
    # print(data_type + '/' +orin_name +".csv")
    print('input file is', inputfile)
    # inputfile2 = inputfile.format(name)
    """############################################################################################"""
    if not os.path.exists(new_folder + "/" + name):  # 判断所在目录下是否有该文件名的文件夹
        os.makedirs(new_folder + "/" + name)
        print("create success")
    """############################################################################################"""
    # sl = 120
    winsize = sl * 2
    num = 0
    for j in inputfile:
        print(j)
        tmp_df = pd.read_csv(j, index_col=False, header=None, usecols=[1])  # open the file
        y = tmp_df.iloc[:, 0]
        filenum = (len(y) // sl)  # decide the number of processing files
        tem = []
        file = []
        with open(j, mode="r", encoding="utf-8") as fp1:  # open the file
            for count in range(filenum):
                print('num is', num)
                num = num + 1
                # text = new_folder + "/save_temp/" + name + "/" + ID + "{0}{1}.csv"
                text = new_folder + "/" + name + "/" + ID + "{0}{1}.csv"

                result = text.format(name + '-', num)
                file.append(result)  # decide the processing file's name
                for line in itertools.islice(fp1, 0, sl):  # every sl'' points, save into tem ist
                    tem.append(line)

            for count in range(filenum):
                with open(file[count], mode="w", encoding="utf-8",
                          newline="\n") as fp2:  # open the final processing file
                    n = count * sl  # determine the saved data
                    if count == (filenum - 1):
                        for i in range(winsize):  # segmetation length is 'winsize'
                            fp2.write(tem[n])  # save
                            n = n + 1
                            if n == (count + 1) * sl:
                                n = 1
                    else:
                        for i in range(winsize):  # segmetation length is 'winsize'
                            fp2.write(tem[n])  # save
                            n = n + 1  # cycling


def function():
    print("Hello world")





