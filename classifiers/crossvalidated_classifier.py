#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 23:16:15 2020

@author: Eli
"""

from sklearn.model_selection import cross_validate
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm
from joblib import dump

import pandas as pd
import numpy as np

class Classifier:
    """
    Simple class for keeping track of data and running
    crossvalidation on sklearn classifiers
    """

    def __init__(self, model):
        self.data = []

        self.set_model(model)

    def set_model(self, model):
        """
        Sets the model for classification

        Parameters
        ----------
        model : sklearn classifier


        Returns
        -------
        None.

        """

        self.model = model

    def load_data(self, data, cols = None):
        """
        Loads data and appends it to the internal dataset

        Parameters
        ----------
        data : pd dataframe or path to load one
            feature matrix, where the column 'label' has the class

        cols : list, optional
            list of columns to keep. If none is given then keeps all

        Returns
        -------
        None.

        """

        if isinstance(data, str):
            data = pd.read_csv(data)

        data_to_append = data.copy()

        #get column subset if needed
        if cols is not None:
            cols_ = cols[:]
            if 'user_id' not in cols:
                cols_.append('user')
            if 'label' not in cols:
                cols_.append('label')
            if 'dataset' not in cols:
                cols_.append('dataset')
            data_to_append = data_to_append[cols_]

        self.data.append(data_to_append)

    def crossval(self, split_col = 'user', cols = None, col_score_split=['user','label']):
        """
        Creates a crossvalidated classifier

        Parameters
        ----------
        split_col : str , optional
            column to perform the crossvalidation over

        cols : list, optional
            list of columns to use in the classifier. If none is given then keeps all
            
        col_score_split: list of str
            list of columns to calculate the score breakdown on

        Returns
        -------
        None.

        """
        #concatenate all of the data together
        if len(self.data) > 1:
            all_data = pd.concat(self.data, axis=0, ignore_index=True, copy=False)
        elif len(self.data) == 1:
            all_data = self.data[0]
        else:
            raise ValueError("I gots no data :'(")

        #select columns
        y = all_data['label'].values
        groups = all_data[split_col].values
        cv = GroupKFold(n_splits=len(np.unique(groups)))
        
        if cols is None:
            cols_ = [c for c in all_data.columns if c not in ['label','dataset','user']]
        else:
            cols_ = cols
        

        X = all_data[cols_].to_numpy()

        print("Beginning model evaluation...")
        # scores = cross_validate(estimator = self.model,
        #                         X = X, y = y, groups=groups,
        #                         cv=cv,
        #                         return_train_score=False,
        #                         return_estimator=True, n_jobs=2)
        
        preds = cross_val_predict(estimator=self.model,
                                  X=X, y=y, groups=groups,
                                  cv=cv, n_jobs=2)

        # scores are in the order of the groups, so the first row out is the
        # result of training on the other groups, and testing on the first group
        #self.scores = scores
        self.preds = preds
        
        #do a score breakdown by unique value
        scores = {}
        for col in col_score_split:
            unique_vals = all_data[col].unique()
            accuracy = np.zeros(len(unique_vals))
            for i,val in enumerate(unique_vals):
                entries = all_data[col] == val
                accuracy[i] = np.sum(self.preds[entries] == y[entries])/np.sum(entries)
                
            scores[col] = pd.DataFrame({col:unique_vals,'accuracy':accuracy})

        return scores
    
    

    def save_crossval_model(self, save_path):
        dump(self.scores, save_path)



def wrapper(path_in, split_col, savePath, col_score_split=['user','label']):
    """
    wrapper for cross val classifier: applies 3 models, to the accerleration, and gyroscope data

    Parameters
    ----------
    path : string or list of str, dataframe or list of dataframe
        path to the csv of interest for running the classifiers.
    split_col: string
        which column to use for cross validataion
    savePath: string
        where to save the csv
    col_score_split: list of str
       list of columns to calculate the score breakdown on

    Returns
    -------
    None.

    """
    if isinstance(path_in, str):
        data = [pd.read_csv(path_in)]
    elif isinstance(path_in, pd.core.frame.DataFrame):
        data = [path_in]
    elif isinstance(path_in, list):
        if isinstance(path_in[0], str):
            data = [pd.read_csv(p) for p in path_in]
        else:
            data = [p for p in path_in]
        
    
    modelList = []
    model = KNeighborsClassifier(n_neighbors=3)
    modelList.append(model)
    model = ExtraTreesClassifier(n_estimators=100)    
    modelList.append(model)
    # model = svm.SVC()
    # modelList.append(model)
    
    modelNames = ['k-NN', 'extra-Trees']#, 'SVC']
    
    scoreDf_list = [pd.DataFrame() for x in col_score_split]
    for i_col, col in enumerate(col_score_split):
        scoreDf_list[i_col][col] = list(data[0][col].unique())+['mean','std']
    
    for idx, model in enumerate(modelList):
        clf = Classifier(model)
        for d_set in data:
            all_feats = d_set.columns
            acc_feats = [f for f in all_feats if 'a_' in f or 'yaw_' in f or 'pitch_' in f or 'roll_' in f]
            clf.load_data(d_set, acc_feats)
            
        scores = clf.crossval(split_col=split_col,col_score_split=col_score_split)
        for i_col, col in enumerate(col_score_split):
            accuracy = scores[col]['accuracy'].values
            scoreDf_list[i_col][modelNames[idx]] = list(accuracy)+[np.mean(accuracy), np.std(accuracy)]
    
    for i in range(len(scoreDf_list)):
        scoreDf_list[i].to_csv(savePath+"_"+col_score_split[i]+'.csv', index = False)
    
    return scoreDf_list
    
if __name__ == "__main__":
    
#     # model = KNeighborsClassifier(n_neighbors=3)
#     model = ExtraTreesClassifier(n_estimators=100)
# #    model = svm.SVC()
    
#     clf = Classifier(model)
    
#     data_path = '/Volumes/GoogleDrive/My Drive/Harvey Mudd/Work/Summer 2020/project_data/MotionSense_FeatMat.csv'
# #    save_path = '/Volumes/GoogleDrive/My Drive/Harvey Mudd/Work/Summer 2020/project_data/results/extra_trees.csv'
#     # save_path = '/Users/kaikaneshina/Documents/MATH178/project_data/UCI_motionSense/K-NN.csv'

#     # data_path1 = '/Users/kaikaneshina/Documents/MATH178/project_data/UCI HAR Dataset/UCI_HAR_FeatMat.csv'

#     data = pd.read_csv(data_path)

#     # data1 = pd.read_csv(data_path1)
#     # # data2 = pd.read_csv(data_path2)
#     # # labels = list(set(data2.label.unique()).intersection(set(data1.label.unique())))
    
#     # # data1 = data1[[True if x in labels else False for x in data1.label]]
#     # # data2 = data2[[True if x in labels else False for x in data2.label]]
    
#     all_feats = data.columns
#     acc_feats = [f for f in all_feats if 'a_' in f]


#     clf.load_data(data, acc_feats)
    
#     # all_feats = data.columns
#     # acc_feats = [f for f in all_feats if 'a_' in f]

#     # clf.load_data(data2, acc_feats)
    
#     scores = clf.crossval(split_col='user')
    
    #clf.save_crossval_model('test.pkl')
    # np.savetxt(save_path,scores['test_score'])
    
    
    # data_path = '/Users/kaikaneshina/Documents/MATH178/project_data/motionSense/MotionSense_FeatMat.csv'
    # save_path = '/Users/kaikaneshina/Documents/GitHub/Colin_Eli_Kai_Math178_Final/results/motionSense/userCrossVal/raw/noGyroRaw128.csv'

    # data_path = '/Users/kaikaneshina/Documents/MATH178/project_data/motionSense/MotionSense_FeatMat_Rotated.csv'
    # save_path = '/Users/kaikaneshina/Documents/GitHub/Colin_Eli_Kai_Math178_Final/results/motionSense/userCrossVal/rotated/rotated128.csv'
    
    # data_path = '/Volumes/GoogleDrive/My Drive/Harvey Mudd/Work/Summer 2020/project_data/Feature_Matrix_128/MotionSense_FeatMat_Rotated.csv'
    data_path = '/Volumes/GoogleDrive/My Drive/Harvey Mudd/Work/Summer 2020/project_data/Feature_Matrix_128/MotionSense_FeatMat.csv'
    data = pd.read_csv(data_path)
    save_path = '/Volumes/GoogleDrive/My Drive/Harvey Mudd/Work/Summer 2020/Colin_Eli_Kai_Math178_Final/results/motionSense/normal_split_by_activity/accuracy'
    
    # # mobiact: regular
    # data_path = '/Users/kaikaneshina/Documents/MATH178/project_data/MobiAct_Dataset_v2.0/mobiAct_FeatMat.csv'
    # save_path = '/Users/kaikaneshina/Documents/GitHub/Colin_Eli_Kai_Math178_Final/results/mobiAct/userCrossVal/raw/raw128.csv'
    # # mobiact: rotated
    # # data_path = '/Users/kaikaneshina/Documents/MATH178/project_data/MobiAct_Dataset_v2.0/mobiAct_FeatMat_Rotated.csv'
    scoreDf_list = wrapper(data_path, 'user', save_path,['user','label'])

