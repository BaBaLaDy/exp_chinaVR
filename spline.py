# -*- coding: utf-8 -*-
"""
The cubic spline to obtain the continous data
including the derivate - first and second order

@author: csxia
"""

import numpy as np
from scipy import interpolate


def np_move_avg(a, n, mode="same"):
    return (np.convolve(a, np.ones((n,)) / n, mode=mode))


class spline_cal:
    def __init__(self, timestep, ini_data, fre):
        '''
        Import: timestep is the initial time point from Unity
                ini_data is the pos data from Unity
                new_fre is the required frequency, unit is Hz
        Export: resampled data
        '''
        self.timestep = timestep
        self.ini_data = ini_data
        self.fre = fre
        self.starttime = timestep[0]
        self.endtime = timestep[timestep.shape[0] - 1]

    def resample(self):

        model = interpolate.CubicSpline(self.timestep, self.ini_data)
        t = []
        for i in range(int((self.endtime - self.starttime) / (1 / self.fre))):
            t.append(self.starttime + float(1 / self.fre) * i)

        re_data = model(t)
        # re_data = np_move_avg(re_data, 20, mode='same')
        return re_data  # , model

    def first_derivate(self):

        model = interpolate.CubicSpline(self.timestep, self.ini_data)
        t = []
        for i in range(int((self.endtime - self.starttime) / (1 / self.fre))):
            t.append(self.starttime + float(1 / self.fre) * i)

        first = interpolate.CubicSpline.derivative(model, 1)
        f_data = first(t)
        # f_data = np_move_avg(f_data, 20, mode='same')

        return f_data

    def second_derivate(self):
        model = interpolate.CubicSpline(self.timestep, self.ini_data)
        t = []
        for i in range(int((self.endtime - self.starttime) / (1 / self.fre))):
            t.append(self.starttime + float(1 / self.fre) * i)

        second = interpolate.CubicSpline.derivative(model, 2)
        s_data = second(t)
        # s_data = np_move_avg(s_data, 30, mode='same')

        return s_data


