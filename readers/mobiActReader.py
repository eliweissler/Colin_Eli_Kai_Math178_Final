#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 21:16:56 2020

@author: kaikaneshina
"""

import pandas as pd
import glob
import os
import numpy as np
from scipy import signal
pd.options.mode.chained_assignment = None  


paths = ['/Users/kaikaneshina/Documents/MATH178/project_data/MobiAct_Dataset_v2.0/Annotated Data/WAL',
'/Users/kaikaneshina/Documents/MATH178/project_data/MobiAct_Dataset_v2.0/Annotated Data/STU',
'/Users/kaikaneshina/Documents/MATH178/project_data/MobiAct_Dataset_v2.0/Annotated Data/STN',
'/Users/kaikaneshina/Documents/MATH178/project_data/MobiAct_Dataset_v2.0/Annotated Data/STD',
'/Users/kaikaneshina/Documents/MATH178/project_data/MobiAct_Dataset_v2.0/Annotated Data/SIT',
'/Users/kaikaneshina/Documents/MATH178/project_data/MobiAct_Dataset_v2.0/Annotated Data/JOG']

newColNames = {'acc_x':'a_x', 'acc_y':'a_y', 'acc_z':'a_z', 'gyro_x':'rot_x', 
           'gyro_y':'rot_y', 'gyro_z':'rot_z', 'azimuth':'yaw'}
newLabels = {'WAL':'wlk', 'STU':'ups', 'STN':'dws', 'STD':'std', 'JOG':'jog'}

numObs = 128
overlap = int(numObs/2)

cols = ['a_x', 'a_y', 'a_z', 'rot_x', 'rot_y', 'rot_z', 'yaw', 'pitch',
       'roll']
num_labels = np.arange(numObs).astype(str)
col_labels = []
for n in num_labels:
    for feat in cols:
        col_labels.append(feat+"_"+n)
        
dfList = []
for folder in paths:
    files = glob.glob(folder + r'/*.csv')
    print(len(files))
    for f in files:
        actFeatureVectors =[]
        data = pd.read_csv(f)
        fname = os.path.basename(f)
        parts = fname.split('_')
        user = int(parts[1])
        lbl = parts[0]
        data.drop(columns = ['rel_time', 'timestamp'], inplace = True)
        # grab only the data based on the folder name
        df = data[data['label']==lbl]
        df.reset_index(inplace = True, drop = True)
        
        df.rename(columns = newColNames, inplace= True)
        
        df['label'] = df['label'].map(newLabels)
        feats = df.columns[df.columns!='label']

        # determine spacing
        spacing = np.arange(0,df.shape[0],overlap)
        # skip the last value in the spacing since we add overlap*2 to each value 
        # for indexing
        for idx in spacing[:-2]:
            subset = df.iloc[idx:idx + numObs][feats]
            featVect = subset.values.flatten()
            actFeatureVectors.append(featVect)
            
 
        actFeatureMatrix = pd.DataFrame(np.array(actFeatureVectors), columns = col_labels)
        actFeatureMatrix['dataset'] = 'mobiAct'
        actFeatureMatrix['user'] = user 
        actFeatureMatrix['label'] = df.label.unique()[0]
        dfList.append(actFeatureMatrix)
        
total = pd.concat(dfList,ignore_index = True,sort=False)
total.to_csv('/Users/kaikaneshina/Documents/MATH178/project_data/MobiAct_Dataset_v2.0/mobiAct_FeatMat.csv',
             index = False)
            
        