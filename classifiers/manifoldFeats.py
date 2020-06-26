#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 19:55:50 2020

@author: kaikaneshina
"""

import pandas as pd
import os

from features import calc_manifold_feats

ma_normal = '/Users/kaikaneshina/Documents/MATH178/project_data/256_data/mobiact/mobiAct_FeatMat_256.csv'
ms_normal = '/Users/kaikaneshina/Documents/MATH178/project_data/256_data/motion_sense/MotionSense_FeatMat_256.csv'
savePath = '/Users/kaikaneshina/Documents/MATH178/project_data/Math189Testing'

csvList = [ma_normal,ms_normal]
for csv in csvList:
    df = pd.read_csv(csv)
    cols = [i for i in df.columns if 'a_' in i]
    
    # do curvature and curvature fft: output it curv, tors, curvFFT, torsFFT
    manifoldFeats = df[cols].apply(calc_manifold_feats,axis = 1)
    # turn rows of lists into df
    newDf = manifoldFeats.apply(pd.Series)
    
    # make the columns based on the features
    dataSize = len(cols)//3
    fftSize = dataSize//2
    newCols = ['curv_' + str(i) for i in range(dataSize)]
    newCols += ['tors_' + str(i) for i in range(dataSize)]
    newCols += ['curvFFT_' + str(i) for i in range(fftSize)]
    newCols += ['torsFFT_' + str(i) for i in range(fftSize)]
    
    newDf.columns = newCols
    
    # add back in the dataset, user, and label info
    newDf[['dataset', 'user', 'label']] = df[['dataset', 'user', 'label']]
    
    # grab file name and create saveFile path 
    fname = os.path.basename(csv)
    saveFile = os.path.join(savePath,fname)
    
    # save out newDf
    newDf.to_csv(saveFile,index = False)
    
    