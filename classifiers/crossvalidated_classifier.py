#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 23:16:15 2020

@author: Eli
"""

from sklearn.model_selection import cross_validate
from sklearn.model_selection import GroupKFold
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
        
    def crossval(self, split_col = 'user', cols = None):
        """
        Creates a crossvalidated classifier

        Parameters
        ----------
        split_col : TYPE, optional
            DESCRIPTION. The default is 'user'.
            
        cols : list, optional
            list of columns to use in the classifier. If none is given then keeps all
        
       
        Returns
        -------
        None.

        """
        #concatenate all of the data together
        if len(self.data) > 1:
            all_data = pd.concat(self.data, axis=0, ignore_axis=True, copy=False)
        elif len(self.data) == 1:
            all_data = self.data[0]
        else:
            raise ValueError("I gots no data :'(")
        
        #select columns
        y = all_data['label'].values
        groups = all_data['user'].values
        
        if cols is None:
            cols_ = [c for c in all_data.columns if c not in ['label','dataset','user']]
        else:
            cols_ = cols
            
        X = all_data[cols_].to_numpy()
        
        print("Beginning model evaluation...")
        scores = cross_validate(estimator = self.model,
                                X = X, y = y, groups=groups,
                                cv=GroupKFold(n_splits=len(np.unique(groups))), 
                                return_train_score=False,
                                return_estimator=True, n_jobs=2)
        
        self.scores = scores
        
        return scores
    
    
    def save_crossval_model(self, save_path):
        dump(scores, save_path)
    
    
    
if __name__ == "__main__":
    
    model = KNeighborsClassifier(n_neighbors=3)
#    model = ExtraTreesClassifier(n_estimators=100)
#    model = svm.SVC()
    
    clf = Classifier(model)
    
#    data_path = '/Volumes/GoogleDrive/My Drive/Harvey Mudd/Work/Summer 2020/project_data/MotionSense_FeatMat.csv'
#    save_path = '/Volumes/GoogleDrive/My Drive/Harvey Mudd/Work/Summer 2020/project_data/results/extra_trees.csv'
    data_path = '/Users/kaikaneshina/Documents/MATH178/project_data/UCI HAR Dataset/UCI_HAR_FeatMat.csv'
    save_path = '/Users/kaikaneshina/Documents/GitHub/Colin_Eli_Kai_Math178_Final/results/UCI_HAR/knn.csv'

    data = pd.read_csv(data_path)
    
    all_feats = data.columns
    acc_feats = [f for f in all_feats if 'a_' in f]

    clf.load_data(data, acc_feats)
    scores = clf.crossval()
    
    #clf.save_crossval_model('test.pkl')
    np.savetxt(save_path,scores['test_score'])


