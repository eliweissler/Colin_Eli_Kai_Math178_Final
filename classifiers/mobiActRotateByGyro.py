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

def rotateFeatMats(featMat, savePath, fname, featLen = 128):
    """
    rotate the feature matrix df rows by roll pitch and yaw
    """
    allFeats = featMat.columns
    accFeats = [f for f in allFeats if 'a_' in f]

    gyroFeats = [f for f in allFeats if 'yaw_' in f or 'pitch_' in f or 'roll_' in f]

    acc = featMat[accFeats].to_numpy()
    gyro = featMat[gyroFeats].to_numpy()
<<<<<<< HEAD

    rows = int(acc.shape[0]*len(featMat[0]))
    cols = int(acc.shape[1]/len(featMat[0]))

=======
    
    
    rows = int(acc.shape[0]*featLen)
    cols = int(acc.shape[1]/featLen)
    
>>>>>>> 55dfb07d127131f5263d6208b41e4b2a03f28d7d
    accReshaped = acc.reshape(rows, cols)
    gyroReshaped = gyro.reshape(rows, cols)

    # rotate acc by gyro data
    accRotated = rotate_to_zero(accReshaped, gyroReshaped)
<<<<<<< HEAD

    accRotated = accRotated.reshape(acc.shape)

    # rotate the rotated data by the pca axis
    rotPCAData, _ = np.apply_along_axis(PCA_rotate_data, 1, accRotated)

    rotPCAData = np.concatenate(rotPCAData).reshape(acc.shape)

=======
    accRotated = accRotated.reshape(acc.shape)
    
    print('gyro rotation done')
    
    # rotate the rotated data by the pca axis
    pcaData = []
    for i in range(accRotated.shape[0]):
        rotPCAData, _ = PCA_rotate_data(accRotated[i,:])
        pcaData.append(rotPCAData)
    
    print('PCA rotation done')
    
    rotPCAData = np.concatenate(pcaData).reshape(acc.shape)
    
>>>>>>> 55dfb07d127131f5263d6208b41e4b2a03f28d7d
    accDf = pd.DataFrame(rotPCAData, columns = accFeats)
    accDf['dataset'] = featMat['dataset']
    accDf['user'] = featMat['user'] 
    accDf['label'] = featMat['label'] 
    accDf.to_csv(os.path.join(savePath,fname), index = False)

    return

if __name__ == '__main__':
<<<<<<< HEAD

    path = '/Users/kaikaneshina/Documents/MATH178/project_data/MobiAct_Dataset_v2.0/mobiAct_FeatMat.csv'
=======
    
    # path = '/Users/kaikaneshina/Documents/MATH178/project_data/MobiAct_Dataset_v2.0/mobiAct_FeatMat.csv'
    # savePath = '/Users/kaikaneshina/Documents/MATH178/project_data/MobiAct_Dataset_v2.0'
    # fname = 'mobiAct_FeatMat_Rotated.csv'
    path = '/Users/kaikaneshina/Documents/MATH178/project_data/motionSense/MotionSense_FeatMat.csv'
    savePath = '/Users/kaikaneshina/Documents/MATH178/project_data/motionSense'
    fname = 'MotionSense_FeatMat_Rotated.csv'


>>>>>>> 55dfb07d127131f5263d6208b41e4b2a03f28d7d
    featMat = pd.read_csv(path)
    rotateFeatMats(featMat, savePath, fname)
