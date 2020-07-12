#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 12:50:46 2020

@author: Eli
"""


from classifiers.features import hand_crafted
import numpy as np
from classifiers.featMatHelpers import getAcc
import pandas as pd
import matplotlib.pyplot as plt
import os



def get_all_hc(fMat):
    """
    Gets the average 

    Parameters
    ----------
    fMat : TYPE
        DESCRIPTION.
    activity : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    hc_feats = []
    n_entries = len(fMat.index)
    for i in range(n_entries):
        if i%1000 == 0:
            print('done with', i, 'out of', n_entries)
        hc_feats.append(hand_crafted(fMat.iloc[[i]]))
    
    columns = ['avgAcc_x','avgAcc_y','avgAcc_z','avgAccNorm',
               'stdAcc_x','stdAcc_y','stdAcc_z','stdAccNorm',
               'absAvgAcc_x','absAvgAcc_y','absAvgAcc_z','absAvgAccNorm',
               'absStdAcc_x','absStdAcc_y','absStdAcc_z','absStdAccNorm',
               'avgMag','stdMag','avgNormAcc','stdNormAcc']

    toReturn =  pd.DataFrame(np.array(hc_feats), columns = columns)
    toReturn['label'] = fMat['label']
    
    return toReturn

def make_activity_plots(hc_data,save_loc):
    
    activities = np.unique(hc_data.label)
    feats = [f for f in hc_data.columns if f != 'label']
    for feat in feats:
        fig, ax = plt.subplots()
        for i in range(len(activities)):
            inds = hc_data['label'] == activities[i]
            mean_val = np.mean(hc_data[inds][feat],axis=0)
            std_val = np.std(hc_data[inds][feat],axis=0)
            plt.errorbar(i,mean_val,std_val, capsize=10, linewidth=2, elinewidth=2)
            plt.xticks(ticks=[0,1,2,3,4,5],labels=activities)
        
        plt.savefig(os.path.join(save_loc,feat+'.png'))
        plt.close(fig)
        
    plt.close('all')
 
        
if __name__ == '__main__':
    ms_path = '/Volumes/GoogleDrive/My Drive/Harvey Mudd/Work/Summer 2020/project_data/Feature_Matrices/256_data/motion_sense/MotionSense_FeatMat_256.csv'
    ma_path = '/Volumes/GoogleDrive/My Drive/Harvey Mudd/Work/Summer 2020/project_data/Feature_Matrices/256_data/mobiact/mobiAct_FeatMat_256.csv'

    ms_data = pd.read_csv(ms_path)
    ma_data = pd.read_csv(ma_path)

    #set font
    import matplotlib
    matplotlib.rcParams.update({'font.size': 16})

    hc_ms = get_all_hc(ms_data)
    hc_ma = get_all_hc(ma_data)
    save_ms = '/Volumes/GoogleDrive/My Drive/Harvey Mudd/Work/Summer 2020/Colin_Eli_Kai_Math178_Final/visualizing_data/pics/hc_feats/ms'
    save_ma = '/Volumes/GoogleDrive/My Drive/Harvey Mudd/Work/Summer 2020/Colin_Eli_Kai_Math178_Final/visualizing_data/pics/hc_feats/ma'
    make_activity_plots(hc_ms,save_ms)
    make_activity_plots(hc_ma,save_ma)
