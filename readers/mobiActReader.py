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


# root = '/Users/collopa/Desktop/nonlinear/project/data/mobiact/'
root = '/Volumes/GoogleDrive/My Drive/Harvey Mudd/Work/Summer 2020/project_data/Raw_Data/MobiAct_Dataset_v2.0/Annotated Data'

paths = [os.path.join(root,'WAL'),
         os.path.join(root,'STU'),
         os.path.join(root,'STN'),
         os.path.join(root,'STD'),
         os.path.join(root,'SIT'),
         os.path.join(root,'JOG')]

newColNames = {'acc_x':'a_x', 'acc_y':'a_y', 'acc_z':'a_z', 'gyro_x':'rot_x', 
           'gyro_y':'rot_y', 'gyro_z':'rot_z', 'azimuth':'yaw'}
newLabels = {'WAL':'wlk', 'STU':'ups', 'STN':'dws', 'STD':'std', 'JOG':'jog', 'SIT':'sit'}

numObs = 256
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

        df[['a_x', 'a_y', 'a_z']] = df[['a_x', 'a_y', 'a_z']]/9.81
        
        df[['yaw', 'pitch', 'roll']] = df[['yaw', 'pitch', 'roll']]*(np.pi/180) %(2*np.pi)

        df['label'] = df['label'].map(newLabels)
        feats = df.columns[df.columns!='label']

        diff = df.shape[0]%87
        idxs = np.arange(0,df.shape[0]-diff,87)
        
        vals = df[feats].to_numpy()
        resamp = np.concatenate([signal.resample(vals[i:i+87,:],50) for i in idxs])
        dfNew = pd.DataFrame(resamp, columns = feats)
        
        if df.shape[0] > 115:
            dfNew = dfNew.iloc[115:,:]
            dfNew.reset_index(inplace = True, drop = True)
    
            # determine spacing
            spacing = np.arange(0,dfNew.shape[0],overlap)
            # skip the last value in the spacing since we add overlap*2 to each value 
            # for indexinga
            if len(spacing)>4:
                for idx in spacing[:-2]:
                    subset = dfNew.iloc[idx:idx + numObs][feats]
                    featVect = subset.values.flatten()
                    actFeatureVectors.append(featVect)
                    
        
                actFeatureMatrix = pd.DataFrame(np.array(actFeatureVectors), columns = col_labels)
                actFeatureMatrix['dataset'] = 'mobiAct'
                actFeatureMatrix['user'] = 'mobiAct_' + str(user)
                actFeatureMatrix['label'] = newLabels[lbl]
                dfList.append(actFeatureMatrix)


total = pd.concat(dfList,ignore_index = True,sort=False)


# out = '/Users/collopa/Desktop/nonlinear/project/data/mobiAct_FeatMat_256.csv'
out = '/Volumes/GoogleDrive/My Drive/Harvey Mudd/Work/Summer 2020/project_data/Feature_Matrices/'+str(numObs)+'_data/mobiact/mobiAct_FeatMat_'+str(numObs)+'.csv'
total.to_csv(out,
             index = False)
            
        