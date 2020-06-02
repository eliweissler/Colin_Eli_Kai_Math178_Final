#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 20:20:08 2020

@author: kaikaneshina
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def linearFit(row, X):
    """
    Does a linear fit to each row of the data, returns the difference between
    the point and its prediction.
    input:
        row: single row of dataframe. Assumes that the row is all of the same feature
        X: linearly spaced values that correspond to the data in time
            should be the same length as the row.
    """

    # compute linear fit.
    reg = LinearRegression().fit(X, row)



if __name__ == '__main__':
    data_path2 = '/Users/kaikaneshina/Documents/MATH178/project_data/motionSense/MotionSense_FeatMat.csv'
    data = pd.read_csv(data_path2)

    for act in data.label.unique():
        data2 = data[data['label']==act]
        data2.reset_index(inplace = True)
        all_feats = data2.columns
        acc_feats = [f for f in all_feats if 'a_x_' in f]


        all_feats_g = data2.columns
        acc_feats_g = [f for f in all_feats_g if 'g_x_' in f]

        Y = data2[acc_feats].to_numpy()
        Y_g = data2[acc_feats_g].to_numpy()
        #Y[0,:]

        # make X values based on sample rate and number of points in each row
        # reshape which is necessary for scikit learn
        X = np.linspace(0,(Y.shape[1]-1)*.02, Y.shape[1]).reshape((-1, 1))

        row = Y[0,:] + Y_g[0,:]
        reg = LinearRegression().fit(X, row)
        preds = reg.predict(X)

        plt.figure();
        plt.plot(X,row, label = 'combined')
        plt.plot(X,Y[0,:], label = 'a_x')
        plt.plot(X,Y_g[0,:], label = 'gravity')
        plt.plot(X, preds, label = 'preds')
        plt.title('activity: ' + act)
        plt.legend()
        plt.figure();
        plt.plot(np.fft.fftshift(np.fft.fftfreq(row.shape[0]))*50, abs(np.fft.fftshift(np.fft.fft(row))), label = 'fft')
        plt.title(act)
