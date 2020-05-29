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

from scipy import interpolate
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.integrate import simps 

test_path = '/Users/collopa/Desktop/nonlinear/project/data/motion_sense/MotionSense_FeatMat.csv'

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


#spline the accleration, aka alpha'
t = np.linspace(0, 2.56, 128) #define time interval
a_x_spline = IUS(t, xline)
a_y_spline = IUS(t, yline)
a_z_spline = IUS(t, zline)

#find the velocity, aka alpha
v_x_spline = a_x_spline.antiderivative()
v_y_spline = a_y_spline.antiderivative()
v_z_spline = a_z_spline.antiderivative()

#find alpha''
ap_x_spline = a_x_spline.derivative()
ap_y_spline = a_y_spline.derivative()
ap_z_spline = a_z_spline.derivative()

#find alpha'''
app_x_spline = ap_x_spline.derivative()
app_y_spline = ap_y_spline.derivative()
app_z_spline = ap_z_spline.derivative()

#make a finer time vector
newt = np.linspace(min(t), max(t), num = 20*len(t))

#calculate torsion
torL = []
for tt in newt:
    #calculate vector
    v_1 = np.array([a_x_spline(tt), a_y_spline(tt), a_z_spline(tt)])
    v_2 = np.array([ap_x_spline(tt), ap_y_spline(tt), ap_z_spline(tt)])
    v_3 = np.array([app_x_spline(tt), app_y_spline(tt), app_z_spline(tt)])
    
    #calculate the torsion and shove it in list
    torsion = 3 * np.linalg.det([v_1, v_2, v_3])  \
               /np.linalg.norm(np.cross(v_1, v_2))
    torL += [torsion]


#calculate torsion
curvL = []
for tt in newt:
    #calculate vector
    v_1 = np.array([a_x_spline(tt), a_y_spline(tt), a_z_spline(tt)])
    v_2 = np.array([ap_x_spline(tt), ap_y_spline(tt), ap_z_spline(tt)])
    
    #calculate curvature
    curvature = 2 * np.linalg.norm(np.cross(v_1, v_2)) \
                /np.linalg.norm(v_1)**(3/2)
    curvL += [curvature]




#####################
## MAKE Curvature PLOT
#####################
fig = plt.figure()
plt.xlabel('Time', size = 14)
plt.ylabel('Curvature Magnitude' , size = 14)
plt.title('Motion Sense Sit', size = 20)
plt.plot(newt, curvL)




#####################
## MAKE TORSION PLOT
#####################
# fig = plt.figure()
# plt.xlabel('Time', size = 14)
# plt.ylabel('Torsion Magnitude' , size = 14)
# plt.title('Motion Sense Sitting', size = 20)
# plt.plot(newt, torL)




#####################
## MAKE VELOCITY PLOT
#####################
   
# fig = plt.figure()
# ax = plt.axes(projection='3d')

# final_x, final_y, final_z = v_x_spline(newt), v_y_spline(newt), v_z_spline(newt)
# ax.scatter3D(final_x, final_y, final_z,c=np.arange(len(final_x)),cmap='jet',s=0.5)

#####################
## MAKE Accleration PLOT
#####################

###Uncomment for accleration sphere
# u, v = np.mgrid[0:2*np.pi:200j, 0:np.pi:100j]
# x = np.cos(u)*np.sin(v)
# y = np.sin(u)*np.sin(v)
# z = np.cos(v)
# ax.plot_surface(x, y, z, color="k", alpha = .1)
# ax.plot3D(xline, yline, zline, 'gray')
# ax.scatter3D(xline, yline, zline,c=np.arange(len(xline)),cmap='jet',s=0.5)


plt.show()
 
