#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 15:33:31 2020

@author: Eli
"""

###Copied and edited slightly from example given on 
###https://github.com/mmalekzadeh/motion-sense?fbclid=IwAR3ZBitp8c6ZBxu2Fk43nslZazddUGblRVtrxL1u4co9o-zjTF7mdHCVsS8

import numpy as np
import pandas as pd
import os
##_____________________________

#change this
data_path = '/Volumes/GoogleDrive/My Drive/Harvey Mudd/Work/Summer 2020/motion-sense/data'

def get_ds_infos():
    ## 0:Code, 1:Weight, 2:Height, 3:Age, 4:Gender
    dss = np.genfromtxt(os.path.join(data_path, 'data_subjects_info.csv'),delimiter=',')
    dss = dss[1:]
    print("----> Data subjects information is imported.")
    return dss
##____________

def creat_time_series(num_features, num_act_labels, num_gen_labels, label_codes, trial_codes):
    dataset_columns = num_features+num_act_labels+num_gen_labels
    ds_list = get_ds_infos()
    train_data = np.zeros((0,dataset_columns))
    test_data = np.zeros((0,dataset_columns))
    for i, sub_id in enumerate(ds_list[:,0]):
        for j, act in enumerate(label_codes):
            for trial in trial_codes[act]:
                fname = 'A_DeviceMotion_data/'+act+'_'+str(trial)+'/sub_'+str(int(sub_id))+'.csv'
                raw_data = pd.read_csv(os.path.join(data_path,fname))
                raw_data = raw_data.drop(['Unnamed: 0'], axis=1)
                unlabel_data = raw_data.values
                label_data = np.zeros((len(unlabel_data), dataset_columns))
                label_data[:,:-(num_act_labels + num_gen_labels)] = unlabel_data
                label_data[:,label_codes[act]] = 1
                label_data[:,-(num_gen_labels)] = int(ds_list[i,0])
                ## We consider long trials as training dataset and short trials as test dataset
                if trial > 10:
                    test_data = np.append(test_data, label_data, axis = 0)
                else:    
                    train_data = np.append(train_data, label_data, axis = 0)
    return train_data , test_data
#________________________________


print("--> Start...")
## Here we set parameter to build labeld time-series from dataset of "(A)DeviceMotion_data"
num_features = 12 # attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)
num_act_labels = 6 # dws, ups, wlk, jog, sit, std
num_gen_labels = 1 # user id
label_codes = {"dws":num_features, "ups":num_features+1, "wlk":num_features+2, "jog":num_features+3, "sit":num_features+4, "std":num_features+5}
trial_codes = {"dws":[1,2,11], "ups":[3,4,12], "wlk":[7,8,15], "jog":[9,16], "sit":[5,13], "std":[6,14]}    
## Calling 'creat_time_series()' to build time-series
print("--> Building Training and Test Datasets...")
train_ts, test_ts = creat_time_series(num_features, num_act_labels, num_gen_labels, label_codes, trial_codes)
print("--> Shape of Training Time-Seires:", train_ts.shape)
print("--> Shape of Test Time-Series:", test_ts.shape)



column_names = ['roll', 'pitch', 'yaw',
                'g_x', 'g_y', 'g_z',
                'rot_x', 'rot_y', 'rot_z',
                'a_x', 'a_y', 'a_z',
                'dws', 'ups', 'wlk', 'jog',
                'sit', 'std','user']


#make it into a pandas dataframe

train_ts = pd.DataFrame(data = train_ts, index = np.arange(train_ts.shape[0]),
                        columns=column_names)


test_ts = pd.DataFrame(data = train_ts, index = np.arange(train_ts.shape[0]),
                        columns=column_names)


#make column for activity labels
activity_train = np.full(len(train_ts), "", dtype=str)
activity_train[train_ts['dws'].astype(bool)] = 'dws'
activity_train[train_ts['ups'].astype(bool)] = 'ups'
activity_train[train_ts['wlk'].astype(bool)] = 'wlk'
activity_train[train_ts['jog'].astype(bool)] = 'jog'
activity_train[train_ts['sit'].astype(bool)] = 'sit'
activity_train[train_ts['std'].astype(bool)] = 'std'


activity_test = np.full(len(test_ts), "", dtype=str)



