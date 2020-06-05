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

def rotateFeatMats(featMat, savePath, fname, featLen = 256, just_pca = True):
    """
    rotate the feature matrix df rows by roll pitch and yaw
    """
    
    # pick out the acceleration and gyroscope data
    acc = getAcc(featMat)
    gyro = getYPR(featMat)
    
    if not just_pca:

        # rotate acc using gyro data. Only rotate the unique elements 
        # to avoid duplicating work on the overlap
        if featMat.shape[0] > 1:
            acc_gyro, inverse = np.unique(np.hstack((acc,gyro)), return_inverse=True, axis=0)    
            acc_unique_rotated = rotate_to_zero(acc_gyro[:,:3], acc_gyro[:,3:])
            accRotated = acc_unique_rotated[inverse]
        else:
            accRotated = rotate_to_zero(acc, gyro)
    
        accRotated = accRotated.reshape(-1,featLen*3)
        
    else:
        accRotated = acc.reshape(-1,featLen*3)
        
    
    print('gyro rotation done')
    
    # rotate the rotated data by the pca axis
    pcaData = []
    new_coords = []
    for i in range(accRotated.shape[0]):
        rotPCAData, axes = PCA_rotate_data(accRotated[i,:], n_points = featLen)
        pcaData.append(rotPCAData)
        new_coords.append(axes)
    
    print('PCA rotation done')
    
    rotPCAData = np.concatenate(pcaData).reshape(-1,featLen*3)
    
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
    path256 = '/Volumes/GoogleDrive/My Drive/Harvey Mudd/Work/Summer 2020/project_data/Feature_Matrices/256_data'
    path128 = '/Volumes/GoogleDrive/My Drive/Harvey Mudd/Work/Summer 2020/project_data/Feature_Matrices/128_data'
    ##MotionSense256
    # ms256 = {'path' : os.path.join(path256,'MotionSense_FeatMat_256.csv'),
    #           'savePath': path256,
    #           'fname':'MotionSense_FeatMat_256_Rotated.csv',
    #           'featLen':256
    #           }
    
    # ma256 = {'path': os.path.join(path256,'mobiAct_FeatMat_256.csv'),
    #           'savePath': path256,
    #           'fname': 'mobiAct_FeatMat_256_Rotated.csv',
    #           'featLen':256
    #           }
    
    
    # ##MotionSense128
    # ms128 = {'path' : os.path.join(path128,'MotionSense_FeatMat.csv'),
    #          'savePath': path128,
    #          'fname': 'MotionSense_FeatMat_Rotated.csv',
    #          'featLen':128
    #          }
    
    # ma128 = {'path' : os.path.join(path128,'mobiAct_FeatMat.csv'),
    #          'savePath':path128,
    #          'fname':'mobiAct_FeatMat_Rotated.csv',
    #          'featLen':128
    #          }
    
    ms256 = {'path' : os.path.join(path256,'motion_sense','MotionSense_FeatMat_256.csv'),
              'savePath': os.path.join(path256,'motion_sense'),
              'fname':'MotionSense_FeatMat_256_PCA.csv',
              'featLen':256
              }
    
    ma256 = {'path': os.path.join(path256,'mobiact','mobiAct_FeatMat_256.csv'),
              'savePath': os.path.join(path256,'mobiact'),
              'fname': 'mobiAct_FeatMat_256_PCA.csv',
              'featLen':256
              }
    
    
    ##MotionSense128
    ms128 = {'path' : os.path.join(path128,'motion_sense','MotionSense_FeatMat.csv'),
             'savePath': os.path.join(path128,'motion_sense'),
             'fname': 'MotionSense_FeatMat_PCA.csv',
             'featLen':128
             }
    
    ma128 = {'path' : os.path.join(path128,'mobiact','mobiAct_FeatMat.csv'),
             'savePath':os.path.join(path128,'mobiact'),
             'fname':'mobiAct_FeatMat_PCA.csv',
             'featLen':128
             }
    
    
    # to_run = [ms128]
    # to_run = [ms128, ma128, ma256, ms256]
    to_run = [ma256, ms256]
    
    for f in to_run:
        featMat = pd.read_csv(f['path'])
        rotateFeatMats(featMat, f['savePath'], f['fname'], f['featLen'],just_pca=True)
