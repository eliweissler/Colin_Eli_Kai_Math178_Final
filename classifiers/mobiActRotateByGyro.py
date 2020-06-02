#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 16:45:56 2020

@author: kaikaneshina
"""

from quaternions import rotate_to_zero, PCA_rotate_data
import pandas as pd
import os
import numpy as np

def rotateFeatMats(featMat, savePath, fname):
    """
    rotate the feature matrix df rows by roll pitch and yaw
    """
    allFeats = featMat.columns
    accFeats = [f for f in allFeats if 'a_' in f]

    gyroFeats = [f for f in allFeats if 'yaw_' in f or 'pitch_' in f or 'roll_' in f]

    acc = featMat[accFeats].to_numpy()
    gyro = featMat[gyroFeats].to_numpy()

    rows = int(acc.shape[0]*len(featMat[0]))
    cols = int(acc.shape[1]/len(featMat[0]))

    accReshaped = acc.reshape(rows, cols)
    gyroReshaped = gyro.reshape(rows, cols)

    # rotate acc by gyro data
    accRotated = rotate_to_zero(accReshaped, gyroReshaped)

    accRotated = accRotated.reshape(acc.shape)

    # rotate the rotated data by the pca axis
    rotPCAData, _ = np.apply_along_axis(PCA_rotate_data, 1, accRotated)

    rotPCAData = np.concatenate(rotPCAData).reshape(acc.shape)

    accDf = pd.DataFrame(rotPCAData, columns = accFeats)
    accDf.to_csv(os.path.join(savePath,fname), index = False)

    return

if __name__ == '__main__':

    path = '/Users/kaikaneshina/Documents/MATH178/project_data/MobiAct_Dataset_v2.0/mobiAct_FeatMat.csv'
    featMat = pd.read_csv(path)
    savePath = '/Users/kaikaneshina/Documents/MATH178/project_data/MobiAct_Dataset_v2.0'
    fname = 'mobiAct_FeatMat_Rotated.csv'
    rotateFeatMats(featMat, savePath, fname)
