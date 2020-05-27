# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

trainPath = '/Users/kaikaneshina/Documents/MATH178/project_data/UCI HAR Dataset/train'
testPath = '/Users/kaikaneshina/Documents/MATH178/project_data/UCI HAR Dataset/test'


lbl = {1: 'wlk', 2:'ups', 3: 'dws', 4:'sit', 5:'std', 6:'lay'}

pathList = [trainPath, testPath]

allDfs = []
for path in pathList:
    folder = path.split('/')[-1]
    users = pd.read_csv(os.path.join(path, 'subject_' + folder + '.txt'), header = None, names = ['users'])
    users = users.loc[:,'users'].to_list()

    activities = pd.read_csv(os.path.join(path, 'y_' + folder + '.txt'), header = None, names = ['acts'])

    files = glob.glob(os.path.join(path, 'Inertial Signals','*.txt'))
    dfList = []
    for f in files:
        colName = os.path.basename(f).split('_')
        if colName[1]=='acc':
            name = 'a_' + colName[2]
        elif colName[1]=='gyro':
            name = 'rot_' + colName[2]
        else:
            name = 'total_' + colName[2] 
        if colName[0] != 'total':
            cols = [name + '_' + str(i) for i in range(128)]
            dataType = {}
            for c in cols:
                dataType[c] = float
            # replace two spaces with 1 spaces in each file then change this line to " "
            df = pd.read_csv(f, header = None, names = cols, delimiter = " ")
            # take only the first 64 values, to avoid the overlaps
            # transpose so we can flatten across the columns aka old rows
            subset = df.iloc[:,:64].T
            # flatten into a single column
            combined = pd.melt(subset)
            # delete the variable column, which contains the old column names
            combined = combined.drop(columns = ['variable'])
            # rename the column based on the data
            combined.columns = [name]
            dfList.append(combined)
    # combined the dataframe list
    dfTotal = pd.concat(dfList, sort = False, axis= 1)
    
    # replace the activity number with the names of the activity
    acts = activities.loc[:,'acts'].map(lbl).to_list()
    # expand the number of activities/users to be the same as the amount of data
    dfTotal['activity'] = np.array(acts*64).reshape(64, len(acts)).flatten('F')
    dfTotal['users'] = np.array(users*64).reshape(64, len(users)).flatten('F')

    # save total df 
    dfTotal.to_csv(os.path.join(path, folder + '_set.csv'), index = False)
    
    allDfs.append(dfTotal)
comb = pd.concat(allDfs, sort = False, ignore_index = True)
comb.to_csv(os.path.join(path, 'allData.csv'), index = False)
        
