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



    
def createFeatVects(data, feats, numObs, overlap, dataset_name, 
                    rotate_to_zero = False):
    """
    Creates a feature matrix from a raw observation csv

    Parameters
    ----------
    data : pd dataframe
        a pandas dataframe where each row is an observation.
        Sampling rate is assumed to be 50 hz.
    feats : list
        list of column names from data to include
    numObs : int
        how many observations should be included in a single row of the 
        feature matrix (i.e. a feature vector)
    overlap : int
        how many observations should overlap between adjacent feature vectors
    dataset_name: str
        what is the dataset name?
    save_path : str or path
        path to save the feature matrix
    rotate_to_zero: Bool
        whether or not to rotate the acceleration axes to pitch/roll/yaw = 0

    Returns
    -------
    actFeatureMatrix : pd dataframe
        matrix where each row is a series of observations to be fed into a classifier, i.e.
        [ax0, ay0, az0. ax1, ay1, az1, ..... ax127, ay127, az127]

    """
    
    # activities
    acts = data.activity.unique()
    # users
    users = data.user.unique()
    
    
    # create dictionary with list for each activity matrix
    actFeatureVectors =[]
    actFeatureLabels = []
    userLabels = []
    
    #make column labels
    num_labels = np.arange(numObs).astype(str)
    col_labels = []
    for n in num_labels:
        for feat in feats:
            col_labels.append(feat+"_"+n)
    
    for act in acts:
        # segment data based on the activity for the users
        df = data[data['activity']==act]
        # want the data to be time related, so do by user, make sure the 
        # timestamps are close
        for user in users:
            # create a user df based on the user and activier
            userDF = df[df['user']==user]

            # determine spacing
            spacing = np.arange(0,userDF.shape[0],overlap)
            # skip the last value in the spacing since we add overlap*2 to each value 
            # for indexing
            for idx in spacing[:-2]:
                subset = userDF.iloc[idx:idx + numObs][feats]
                featVect = subset.values.flatten()
                actFeatureVectors.append(featVect)
                actFeatureLabels.append(act)
                userLabels.append(dataset_name+"_"+str(user))
                
                
    actFeatureMatrix = pd.DataFrame(np.array(actFeatureVectors), columns = col_labels)
    actFeatureMatrix['dataset'] = [dataset_name]*len(actFeatureMatrix)
    actFeatureMatrix['user'] = userLabels
    actFeatureMatrix['label'] = actFeatureLabels 
    
    # fix gravity in the createFeatVectors file for motion sense.
    if dataset_name == 'MotionSense':
        allFeats = actFeatureMatrix.columns
        accFeats = [f for f in allFeats if 'a_' in f]
        for val in accFeats:
            actFeatureMatrix[val] += actFeatureMatrix[val.replace('a','g')]
        
    
    return actFeatureMatrix

    


if __name__ == "__main__":
    
    # user defined param: number of x, y, z pairs to put in a single vector
    numObs = 256
    overlap = numObs//2    

    ##MOTIONSENSE DATASET
#    data_path = '/Volumes/GoogleDrive/My Drive/Harvey Mudd/Work/Summer 2020/project_data/'
    #path_path = '/Users/kaikaneshina/Documents/MATH178/project_data'
#    output_path = '/Volumes/GoogleDrive/My Drive/Harvey Mudd/Work/Summer 2020/project_data/MotionSense_FeatMat.csv'
#    test_path = os.path.join(data_path,'test_set.csv')
#    train_path = os.path.join(data_path,'train_set.csv')
#    train_data = pd.read_csv(train_path)
#    test_data = pd.read_csv(test_path)
#    feats = ['roll', 'pitch', 'yaw', 'g_x', 'g_y', 'g_z', 'rot_x', 'rot_y', 'rot_z',
#           'a_x', 'a_y', 'a_z']

    
    ## UCI HAR dataset
    train_path = '/Users/kaikaneshina/Documents/MATH178/project_data/UCI HAR Dataset/train/train_set.csv'
    test_path = '/Users/kaikaneshina/Documents/MATH178/project_data/UCI HAR Dataset/test/test_set.csv'
    output_path = '/Users/kaikaneshina/Documents/MATH178/project_data/UCI HAR Dataset/UCI_HAR_FeatMat.csv'
    train_data = pd.read_csv(train_path)
    train_data = train_data.rename(columns = {'users':'user'})
    test_data = pd.read_csv(test_path)
    test_data = test_data.rename(columns = {'users':'user'})
    cols = test_data.columns.to_list()
    cols.remove('activity')
    cols.remove('user')
    feats = cols
    
    out = []
    for data in [train_data, test_data]:
        out.append(createFeatVects(data, feats, numObs, overlap, 'MotionSense'))
        
    out = pd.concat(out)
    out.to_csv(output_path)
    
    
    # lbl = {'dws': 0, 'ups':1, 'wlk':2, 'jog':3, 'sit':4, 'std':5}
    # data['label'] = data['activity'].map(lbl)
    
    # features
    
    
    
    
    
    
    