#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 21:51:07 2020

@author: kaikaneshina
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def calc_fft(feature_vec):
    """
    takes in feature vector, splits it into its 3 components, and does the fft of each
    returns magnitude of fft for each component 
    """
    # separate components
    xline = feature_vec[0::3]
    yline = feature_vec[1::3]
    zline = feature_vec[2::3]
    
    # do fft of each
    xFFT = np.abs(np.fft.fftshift(np.fft.fft(xline)))
    yFFT = np.abs(np.fft.fftshift(np.fft.fft(yline)))
    zFFT = np.abs(np.fft.fftshift(np.fft.fft(zline)))

    return xFFT, yFFT, zFFT

if __name__ == '__main__':
    # data_path2 = '/Users/kaikaneshina/Documents/MATH178/project_data/motionSense/MotionSense_FeatMat.csv'
    data_path2 = '/Users/collopa/Desktop/nonlinear/project/data/motion_sense/MotionSense_FeatMat.csv'
    data = pd.read_csv(data_path2)
    
    for act in data.label.unique():
        data2 = data[data['label']==act]
        data2.reset_index(inplace = True)
        all_feats = data2.columns
        acc_feats = [f for f in all_feats if 'a_' in f]
    
    
        all_feats_g = data2.columns
        acc_feats_g = [f for f in all_feats_g if 'g_' in f]
    
        Y = data2[acc_feats].to_numpy()
        Y_g = data2[acc_feats_g].to_numpy()
        #Y[0,:]
        
        row = Y[0,:] + Y_g[0,:]
        # make X values based on sample rate and number of points in each row
        # reshape which is necessary for scikit learn
        
        xFFT, yFFT, zFFT = calc_fft(row)
        
        plt.figure();
        plt.plot(np.fft.fftshift(np.fft.fftfreq(int(row.shape[0]/3)))*50, xFFT, label = 'x fft')
        plt.title('activity: ' + act + ' x fft')
        plt.legend()
        
        plt.figure();
        plt.plot(np.fft.fftshift(np.fft.fftfreq(int(row.shape[0]/3)))*50, yFFT, label = 'y fft')
        plt.title('activity: ' + act + ' y fft')
        plt.legend()

        plt.figure();
        plt.plot(np.fft.fftshift(np.fft.fftfreq(int(row.shape[0]/3)))*50, zFFT, label = 'z fft')
        plt.title('activity: ' + act + ' z fft')
        plt.legend()

        plt.show()
