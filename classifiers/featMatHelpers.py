#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 17:18:24 2020

@author: Eli
"""

import numpy as np
import pandas as pd

def getAcc(allFeats):
    """
    

    Parameters
    ----------
    FeatMat : TYPE
        DESCRIPTION.

    Returns
    -------
    nx3 numpy array of acceleration

    """
    a_x = allFeats[[f for f in allFeats.columns if 'a_x' in f]].to_numpy().reshape(-1,1)
    a_y = allFeats[[f for f in allFeats.columns if 'a_y' in f]].to_numpy().reshape(-1,1)
    a_z = allFeats[[f for f in allFeats.columns if 'a_z' in f]].to_numpy().reshape(-1,1)
    
    return np.hstack((a_x,a_y,a_z))
    

def getYPR(allFeats):
    """
    

    Parameters
    ----------
    FeatMat : TYPE
        DESCRIPTION.

    Returns
    -------
    nx3 numpy array of yaw, pitch, roll

    """
    yaw = allFeats[[f for f in allFeats.columns if 'yaw_' in f]].to_numpy().reshape(-1,1)
    pitch = allFeats[[f for f in allFeats.columns if 'pitch_' in f]].to_numpy().reshape(-1,1)
    roll = allFeats[[f for f in allFeats.columns if 'roll_' in f]].to_numpy().reshape(-1,1)
    
    return np.hstack((yaw, pitch,roll))