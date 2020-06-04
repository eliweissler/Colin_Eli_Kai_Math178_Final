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

from featMatHelpers import getAcc, getYPR

def rotateFeatMats(featMat, savePath, fname, featLen = 256):
    """
    rotate the feature matrix df rows by roll pitch and yaw
    """
    
    
    acc = getAcc(featMat)
    gyro = getYPR(featMat)

    
    # rotate acc by gyro data
    accRotated = rotate_to_zero(acc, gyro)

    accRotated = accRotated.reshape(-1,featLen)
    
    print('gyro rotation done')
    
    # rotate the rotated data by the pca axis
    pcaData = []
    new_coords = []
    for i in range(accRotated.shape[0]):
        rotPCAData, axes = PCA_rotate_data(accRotated[i,:])
        pcaData.append(rotPCAData)
        new_coords.append(axes)
    
    print('PCA rotation done')
    
    rotPCAData = np.concatenate(pcaData).reshape(acc.shape)
    
    accFeats = [f for f in featMat.columns if 'a_' in f]
    accDf = pd.DataFrame(rotPCAData, columns = accFeats)
    accDf['dataset'] = featMat['dataset']
    accDf['user'] = featMat['user'] 
    accDf['label'] = featMat['label'] 
    
    if savePath is not None:
        accDf.to_csv(os.path.join(savePath,fname), index = False)

    return accDf, accRotated, new_coords

if __name__ == '__main__':
    
    ###KAI
    ##MobiAct
    # path = '/Users/kaikaneshina/Documents/MATH178/project_data/MobiAct_Dataset_v2.0/mobiAct_FeatMat.csv'
    # savePath = '/Users/kaikaneshina/Documents/MATH178/project_data/MobiAct_Dataset_v2.0'
    # fname = 'mobiAct_FeatMat_Rotated.csv'
    ##MotionSense
    # path = '/Users/kaikaneshina/Documents/MATH178/project_data/motionSense/MotionSense_FeatMat.csv'
    # savePath = '/Users/kaikaneshina/Documents/MATH178/project_data/motionSense'
    # fname = 'MotionSense_FeatMat_Rotated.csv'
    
    ###Eli
    ##MotionSense256
    ms_256 = {'path' : '/Volumes/GoogleDrive/My Drive/Harvey Mudd/Work/Summer 2020/project_data/Feature_Matrices/Feature_Matrix_256/MotionSense_FeatMat_256.csv',
              'savePath':'/Volumes/GoogleDrive/My Drive/Harvey Mudd/Work/Summer 2020/project_data/Feature_Matrices/Feature_Matrix_256',
              'fname':'MotionSense_FeatMat_256_Rotated.csv'}
    
    ##MotionSense128
    ms128 = {'path' : '/Volumes/GoogleDrive/My Drive/Harvey Mudd/Work/Summer 2020/project_data/Feature_Matrices/Feature_Matrix_128/MotionSense_FeatMat.csv',
             'savePath': '/Volumes/GoogleDrive/My Drive/Harvey Mudd/Work/Summer 2020/project_data/Feature_Matrices/Feature_Matrix_128',
             'fname': 'MotionSense_FeatMat_Rotated.csv'}
    
    to_run = [ms128]
    
    for f in to_run:
        featMat = pd.read_csv(f['path'])
        rotateFeatMats(featMat, f['savePath'], f['fname'])
