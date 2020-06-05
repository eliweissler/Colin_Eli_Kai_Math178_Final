#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 13:43:41 2020

@author: Eli
"""

import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pyquaternion import Quaternion


from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation as R

import numpy as np

import scipy

import matplotlib 

from classifiers.quaternions import PCA_rotate_data,rotate_to_zero,rotate_quat
from classifiers.rotateByGyro import rotateFeatMats

from featMatHelpers import getAcc, getYPR

import os

matplotlib.rc('font', size=12)


def normalize_rows(xyz_data):
    """
    Normalizes each row of an nx3 numpy array
    """
    return np.apply_along_axis(lambda x: x/np.linalg.norm(x), axis = 1, arr = xyz_data)


def plot_on_sphere(xyz_data, normalize=False):
    """
    plots each row of an nx3 numpy array on the surface of a sphere
    to do this we first normalize each row
    """
    #make 3d figure
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    #build up a sphere
    u, v = np.mgrid[0:2*np.pi:200j, 0:np.pi:100j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    
    #plot the sphere
    ax.plot_surface(x, y, z, color="k", alpha = .1)
    
    #normalize data then plot it
    if normalize:
        xyz_normalized = normalize_rows(xyz_data)
        ax.scatter3D(xyz_normalized[:,0], xyz_normalized[:,1], xyz_normalized[:,2],s=0.1)
    else:
        ax.scatter3D(xyz_data[:,0], xyz_data[:,1], xyz_data[:,2],s=0.1)
        
    return fig, ax

def plot_adjusted_vs_unadjusted(featVec, savePath, n_entries=256,freq=50,nonlinear=False):
    """
    plots the adjusted and unadjusted acceleration data. FeatVec must be
    a dataframe of length 1, i.e. gotten by FeatMat.iloc[[i]]
    """
    
    
    #create the time axis
    times = np.arange(n_entries)*(1/freq)
    
    #rotate and align
    aligned, rotated, new_coords = rotateFeatMats(featVec, savePath=None, fname=None, featLen = 256)
    
    new_coords = new_coords[0]
    
    acc = getAcc(featVec)
    acc_rotated = rotated.reshape(n_entries, 3)
    acc_align = getAcc(aligned)
    
    # obtain save info:
    act = featVec.label.values[0]
    
    f, ax = plt.subplots(nrows=3,ncols=1, figsize=(25,15) )

    ax[0].plot(times,acc)
    ax[0].legend(['a_x', 'a_y', 'a_z'])
    ax[0].set_xlabel('time (s)')
    ax[0].set_ylabel('acceleration (m/s^2)')
    ax[0].set_title('un-adjusted')
    
    ax[1].plot(times,acc_rotated)
    ax[1].legend(['a_x', 'a_y', 'a_z'])
    ax[1].set_xlabel('time (s)')
    ax[1].set_ylabel('acceleration (m/s^2)')
    ax[1].set_title('rotated')
    
    ax[2].plot(times,acc_align)
    ax[2].legend(['a_x', 'a_y', 'a_z'])
    ax[2].set_xlabel('time (s)')
    ax[2].set_ylabel('acceleration (m/s^2)')
    ax[2].set_title('rotated and aligned')
    f.tight_layout(pad=2)
    plt.savefig(os.path.join(savePath, act + '_all.png'))
    
    
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(acc_rotated[:,0], acc_rotated[:,1], acc_rotated[:,2],s=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    mean_a = np.mean(acc_rotated, axis=0)
    ax.quiver(mean_a[0],mean_a[1],mean_a[2],new_coords[0][0],new_coords[0][1],new_coords[0][2], color = 'r')
    ax.quiver(mean_a[0],mean_a[1],mean_a[2],new_coords[1][0],new_coords[1][1],new_coords[1][2], color = 'b')
    ax.quiver(mean_a[0],mean_a[1],mean_a[2],new_coords[2][0],new_coords[2][1],new_coords[2][2], color = 'g')
    ax.set_title('rotated')
    plt.savefig(os.path.join(savePath, act + '_rotatedVectors.png'))

    
    return 



def plot_activity_heading(featMat,activity, savePath, n_entries=256, downsample = True):
    """
    For each observation plots on a sphere the point defined by the vector
    [1/sqrt(3), 1/sqrt(3), 1/sqrt(3)]
    in the coordinates defined by the roll,pitch, and yaw
    
    plots the roll pitch and yaw data based on the x,y,z basis vectors
    shows the phone's orientation
    
    single x,y,z are where the heading is pointing in the direction 
    
    Returns
    -------
    None.

    """
    # dictionary to translate from activity
    d = {'wlk': 'Walking', 'sit': 'Sitting', 'std': 'Standing', 'ups': 'Walking Up Stairs',
         'dws': 'Walking Down Stairs', 'jog': 'Jogging'}
    
    # grab the activity of interest
    featMat = featMat[featMat['label']==activity]
    
    # downsample the data so it doesn't take forever to run
    if downsample:
        featMat = featMat.loc[::50,:]

    gyro = getYPR(featMat)

    trans = R.from_euler('zxy',gyro).as_quat() #returns the scalar last
    trans = [Quaternion(imaginary = x[:-1], real = x[-1]) for x in trans]
    
    x = np.array([rotate_quat(Quaternion(imaginary = [1,0,0]), rot).imaginary for rot in trans])
    y = np.array([rotate_quat(Quaternion(imaginary = [0,1,0]), rot).imaginary for rot in trans])
    z = np.array([rotate_quat(Quaternion(imaginary = [0,0,1]), rot).imaginary for rot in trans])
    
    # calculate mean / stdev of angles
    # we know that r = 1, as the data is normalized.
    phi = np.rad2deg(np.arctan2(z[:,1], z[:,0]))
    theta = np.rad2deg(np.arccos(z[:,2]))
    
    heading_vec = (x+y+z)/np.sqrt(3)
    
    fig, ax = plot_on_sphere(heading_vec, normalize=False)
    ax.set_title('Activity: ' + d[activity])
    ax.set_xlabel('X', labelpad=10)    
    ax.set_ylabel('Y', labelpad=10)
    ax.set_zlabel('Z', labelpad=10)
    
    plt.savefig(os.path.join(savePath, activity + '.png'))
    
    fig, ax = plot_on_sphere(z, normalize=False)
    ax.set_title('Activity: ' + d[activity])
    ax.set_xlabel('X', labelpad=10)    
    ax.set_ylabel('Y', labelpad=10)
    ax.set_zlabel('Z', labelpad=10)
    
    plt.savefig(os.path.join(savePath, activity + '_z.png'))
    
    return phi, theta
    
    

if __name__ == '__main__':
   
    motionSense = '/Users/kaikaneshina/Documents/MATH178/project_data/featMat256/MotionSense_FeatMat_256.csv'
    mobiAct = '/Users/kaikaneshina/Documents/MATH178/project_data/featMat256/mobiAct_FeatMat_256.csv'
    
    savePath = '/Users/kaikaneshina/Documents/MATH178/project_data/results/256/motionSense'
    
    df = pd.read_csv(motionSense)
    
    
    # users = df.user.unique()[:5]
    # for u in users:
    #     dfSub = df[df['user']==u]
    #     savePathSub = os.path.join(savePath,u)
    #     try: os.mkdir(savePathSub)
    #     except: pass
        
    #     phiList = []
    #     thetaList = []
    #     activities = dfSub.label.unique()
    #     for act in activities:
    #         phi, theta = plot_activity_heading(dfSub,act, savePathSub)
    #         phiList.append( [np.mean(phi), np.std(phi)])
    #         thetaList.append( [np.mean(theta), np.std(theta)])
                         
    #     theta = np.array(thetaList)
    #     phi = np.array(phiList)
        
    #     angles = pd.DataFrame()
    #     angles['activity'] = activities
    #     angles['theta mean'] = theta[:,0]
    #     angles['theta stdev'] = theta[:,1]
    #     angles['phi mean'] = phi[:,0]
    #     angles['phi stdev'] = phi[:,1]
    
    #     angles.to_csv(os.path.join(savePathSub, 'angles.csv'), index = False)
    
    activities = df.label.unique()
    savePath = os.path.join(savePath, 'rotation')
    try: os.mkdir(savePath)
    except: pass
    for act in activities:
        subDf = df[df['label']==act]
        featVect= subDf.iloc[0,:].to_frame().T
        plot_adjusted_vs_unadjusted(featVect, savePath)
    
    
    
    
    
    
   
    