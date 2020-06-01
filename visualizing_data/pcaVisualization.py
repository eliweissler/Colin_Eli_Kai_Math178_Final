#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 19:40:28 2020

@author: kaikaneshina
"""

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import numpy as np
import os
from sklearn.decomposition import PCA 
from mpl_toolkits.mplot3d import proj3d

plt.rcParams.update({'font.size': 22})

def grabCols(df, col):
    """
    grabs columns of the df specified by the stirng (col)
    returns array with the column values in an array
    input:
        df: dataframe
        col: string with name for column
    """
    
    all_feats = df.columns
    feats = [f for f in all_feats if col in f]
    arrVals = df[feats].to_numpy()

    return arrVals

def combineData(df):
    """
    combines gravity and acceleration columns from df
    
    input:
        df: dataframe
    """
    acc = grabCols(df, 'a_')
    grav = grabCols(df, 'g_')
    
    combined = grav + acc
    
    return combined

if __name__ == '__main__':
    data_path2 = '/Users/kaikaneshina/Documents/MATH178/project_data/motionSense/MotionSense_FeatMat.csv'
#    data_path2 = '/Users/collopa/Desktop/nonlinear/project/data/motion_sense/MotionSense_FeatMat.csv'
    
    data = pd.read_csv(data_path2)
    
    for act in data.label.unique():
        data2 = data[data['label']==act]
        data2.reset_index(inplace = True)
        
        combined = combineData(data2)
        # extract the first row
        feature_vec = combined[0,:]
        # reshape the data
        accXYZ = feature_vec.reshape(128,3)
        pca = PCA(n_components = 3)
            
        pca.fit(accXYZ)
        
        # pca.explained_variance_: importance of data on each axis aka their important
        # tells us direction of vector, they are the eigenvalues
        eigVals = pca.explained_variance_
        eigVects = pca.components_
        
        fig = plt.figure(figsize = (20,15))
        ax = fig.add_subplot(111,  projection='3d')
        
        ax.scatter(accXYZ[:, 0], accXYZ[:, 1], accXYZ[:,2], alpha=0.2)
        
        for vector in eigVects:
            # draw vector from mean to the components, aka the eigen vectors
            endPoints = pca.mean_ + vector
            ax.quiver(pca.mean_[0], pca.mean_[1], pca.mean_[2], endPoints[0], endPoints[1], endPoints[2])
        plt.show()











