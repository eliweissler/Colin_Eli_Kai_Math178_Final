#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 10:51:10 2020

@author: kaikaneshina
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle

# this script creates feature vectors based on the activity, and specified overlap
data_path = '/Volumes/GoogleDrive/My Drive/Harvey Mudd/Work/Summer 2020/project_data/'
#path_path = '/Users/kaikaneshina/Documents/MATH178/project_data'
test_path = os.path.join(data_path,'test_set.csv')
train_path = os.path.join(data_path,'train_set.csv')
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

output_path = '/Volumes/GoogleDrive/My Drive/Harvey Mudd/Work/Summer 2020/project_data'


# user defined param: number of x, y, z pairs to put in a single vector
numPairs = 128
overlap = numPairs//2

# lbl = {'dws': 0, 'ups':1, 'wlk':2, 'jog':3, 'sit':4, 'std':5}
# data['label'] = data['activity'].map(lbl)

# features
feats = ['roll', 'pitch', 'yaw', 'g_x', 'g_y', 'g_z', 'rot_x', 'rot_y', 'rot_z',
       'a_x', 'a_y', 'a_z']
num_labels = np.arange(numPairs).astype(str)
col_labels = []
for n in num_labels:
    for feat in feats:
        col_labels.append(feat+"_"+n)

d_sets = {'train':train_data, 'test':test_data}

saveLabels = {'train': 'train_vectors/train',
              'test': 'test_vectors/test'
             }

for d_set in d_sets:
    
    data = d_sets[d_set]
    
    # activities
    acts = data.activity.unique()
    # users
    users = data.user.unique()
    
    
    # create dictionary with list for each activity matrix
    actFeatureVectors =[]
    actFeatureLabels = []
    for act in acts:
        # segment data based on the activity for the users
        df = data[data['activity']==act]
        # want the data to be time related, so do by user, make sure the 
        # timestamps are close
        for user in users:
            # create a user df based on the user and activier
            userDF = df[df['user']==user]
            # use indices to make sure that the data is close in time
            # idxs = userDF.index.values
            # difference in indices
            # diff = idxs[1:] - idxs[:-1]
            # find locations where the difference is not 1 - aka jump in time
            # stopping = np.where(diff!=1)[0]
            # if all of the data is the same in time: (it is for training, I checked)
            # if len(stopping==0):
            # determine spacing
            spacing = np.arange(0,userDF.shape[0],overlap)
            # skip the last value in the spacing since we add overlap*2 to each value 
            # for indexing
            for idx in spacing[:-1]:
                subset = data.loc[idx:idx + numPairs-1,feats]
                featVect = subset.values.flatten()
                actFeatureVectors.append(featVect)
                actFeatureLabels.append(act)
                
    actFeatureMatrix = pd.DataFrame(np.array(actFeatureVectors), columns = col_labels)
    actFeatureMatrix['label'] = actFeatureLabels 
    
    #actFeatureLabels = np.array(actFeatureLabels)
    
    actFeatureMatrix.to_csv(os.path.join(output_path, saveLabels[d_set]+"_featureMat.csv"), index=False)
    #np.savetxt(os.path.join(output_path, saveLabels[d_set]+"_labels.csv"),actFeatureLabels)
    

    