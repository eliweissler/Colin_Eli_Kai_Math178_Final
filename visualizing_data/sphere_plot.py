#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 15:01:38 2020

@author: Eli
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D

test_path = '/Volumes/GoogleDrive/My Drive/Harvey Mudd/Work/Summer 2020/project_data/MotionSense_FeatMat.csv'

#load the matrix
trainMat = pd.read_csv(test_path)


#select rows to use
trainMat_ = trainMat.copy()

#extract labels
train_labels = trainMat_['label'].values
#del trainMat_['label']



#extract appropriate features
all_feats = trainMat_.columns
ax = [f for f in all_feats if 'a_x' in f]
ay = [f for f in all_feats if 'a_y' in f]
az = [f for f in all_feats if 'a_z' in f]


# get the vector for a single vector
user = 'MotionSense_1'
inds = np.logical_and(trainMat_.user == user,trainMat_.label == 'jog')
jogging_ax = trainMat_[inds][ax].to_numpy()
jogging_ay = trainMat_[inds][ay].to_numpy()
jogging_az = trainMat_[inds][az].to_numpy()


inds = np.logical_and(trainMat_.user == user,trainMat_.label == 'sit')
sitting_ax = trainMat_[inds][ax].to_numpy()
sitting_ay = trainMat_[inds][ay].to_numpy()
sitting_az = trainMat_[inds][az].to_numpy()

inds = np.logical_and(trainMat_.user == user,trainMat_.label == 'wlk')
wlk_ax = trainMat_[inds][ax].to_numpy()
wlk_ay = trainMat_[inds][ay].to_numpy()
wlk_az = trainMat_[inds][az].to_numpy()


# first_elem = np.vstack((jogging_ax[20,:], jogging_ay[20,:], jogging_az[20,:])).T
# first_elem = np.vstack((wlk_ax[20,:], wlk_ay[20,:], wlk_az[20,:])).T
first_elem = np.vstack((sitting_ax[20,:], sitting_ay[20,:], sitting_az[20,:])).T
# first_elem = np.vstack((sitting_ax[[20,22,24,26],:].reshape(1,-1), sitting_ay[[20,22,24,26],:].reshape(1,-1), sitting_az[[20,22,24,26],:].reshape(1,-1))).T
# first_elem = np.vstack((jogging_ax[[20,22,24,26],:].reshape(1,-1), jogging_ay[[20,22,24,26],:].reshape(1,-1), jogging_az[[20,22,24,26],:].reshape(1,-1))).T

first_elem_smooth = np.apply_along_axis(lambda x:gaussian_filter(x,2),0,first_elem)
first_elem_norm = np.apply_along_axis(lambda x: x/np.linalg.norm(x),arr=first_elem, axis = 1)
first_elem_norm_smooth = np.apply_along_axis(lambda x: x/np.linalg.norm(x),arr=first_elem_smooth, axis = 1)


# draw sphere

n=len(first_elem)
xline = first_elem_norm_smooth[:n,0]
yline = first_elem_norm_smooth[:n,1]
zline = first_elem_norm_smooth[:n,2]


# try and integrate the velocity
auc = np.cumsum(first_elem_smooth,axis=0)
auc = np.cumsum(auc,axis=0)
xline = auc[:n,0]
yline = auc[:n,1]
zline = auc[:n,2]


#interpolate a continuous path
final_x = []
final_y = []
final_z = []
fill = 50
for i in range(n-1):
    dx = xline[i+1]-xline[i]
    dy = yline[i+1]-yline[i]
    dz = zline[i+1]-zline[i]
    for f in range(fill):
        pt = np.array([xline[i]+dx*f/fill,
                              yline[i]+dy*f/fill,
                              zline[i]+dz*f/fill])
        # pt/=np.linalg.norm(pt)
        final_x.append(pt[0])
        final_y.append(pt[1])
        final_z.append(pt[2])
    
final_x = np.array(final_x)
final_y = np.array(final_y)
final_z = np.array(final_z)
    
    
fig = plt.figure()
ax = plt.axes(projection='3d')
u, v = np.mgrid[0:2*np.pi:200j, 0:np.pi:100j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
# ax.plot_surface(x, y, z, color="k", alpha = .1)
# ax.plot3D(xline, yline, zline, 'gray')
# ax.scatter3D(xline, yline, zline,c=np.arange(len(xline)),cmap='jet',s=0.5)

ax.scatter3D(final_x, final_y, final_z,c=np.arange(len(final_x)),cmap='jet',s=0.5)

plt.xlabel('x')
plt.ylabel('y')
import matplotlib 
matplotlib.rc('font', size=12)

x = np.linspace(0,2.56,128)
plt.figure()
plt.plot(x,first_elem_smooth[:,0])
plt.xlabel('Time (s)')
plt.ylabel('X Acceleration')
plt.figure()
plt.plot(x,first_elem_smooth[:,1], c = 'r')
plt.xlabel('Time (s)')
plt.ylabel('Y Acceleration')
plt.figure()
plt.plot(x,first_elem_smooth[:,2], c = 'k')
plt.xlabel('Time (s)')
plt.ylabel('Z Acceleration')

plt.figure()
plt.plot(np.array([np.linalg.norm(x) for x in first_elem_smooth]))


#plt.close()