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

matplotlib.rc('font', size=12)



# std_4 = pd.read_csv('/Users/Eli/Downloads/MobiAct_Dataset_v2.0/Annotated Data/STD/STD_4_1_annotated.csv')
# std_2 = pd.read_csv('/Users/Eli/Downloads/MobiAct_Dataset_v2.0/Annotated Data/STD/STD_2_1_annotated.csv')
std_4 = pd.read_csv('/Volumes/GoogleDrive/My Drive/Harvey Mudd/Work/Summer 2020/project_data/MobiAct_Dataset_v2.0/Annotated Data/SIT/SIT_10_1_annotated.csv')
plt.figure()
times = (std_4.timestamp - std_4.timestamp.values[0])/(10**9)
plt.plot(times,std_4.pitch%360)
plt.plot(times,std_4.roll%360)
plt.plot(times,std_4.azimuth%360)
plt.ylim((0,360))
plt.xlabel('time (s)')
plt.ylabel('angle (deg)')
plt.legend(['pitch','roll','yaw'])
plt.savefig('mobiact_sitting')

# plt.figure()
# plt.plot(std_2.pitch)
# plt.plot(std_2.roll)
# plt.plot(std_2.azimuth)
# plt.ylim((-np.pi,np.pi))


motion_sense = pd.read_csv('/Volumes/GoogleDrive/My Drive/Harvey Mudd/Work/Summer 2020/project_data/train_set.csv')

std_ms = motion_sense[motion_sense['activity'] == 'jog']
std_ms = std_ms[std_ms['user'] == 2]

a_x = std_ms['a_x']#+std_ms['g_x']
a_y = std_ms['a_y']#+std_ms['g_y']
a_z = std_ms['a_z']#+std_ms['g_z']

acc_ms = np.vstack((a_x, a_y, a_z)).T


ypr_ms = std_ms[['yaw', 'pitch', 'roll']].to_numpy()





def rotate_to_zero(acc, ypr):
    """
    Rotates the acceleration vectors to be in the 
    frame of roll/pitch/yaw of 0

    Parameters
    ----------
    acc : nx3 np array
        [ax, ay, az]
    ypr : nx3 np array
        [yaw pitch roll]

    Returns
    -------
    acc_rot : TYPE
        DESCRIPTION.

    """
    
    #create the rotations to rotate to zero
    trans = R.from_euler('zxy',ypr).as_quat() #returns the scalar last
    trans = [Quaternion(imaginary = x[:-1], real = x[-1]) for x in trans]
    acc_rot = np.array([rotate_quat(Quaternion(imaginary=acc[i,:]), trans[i].conjugate).imaginary for i in range(end_n)])
    
    return acc_rot
    

def plot_adjusted_vs_unadjusted(acc,ypr,freq,end_t,acc_mult = 9.8,nonlinear=False):
    
    
   
    #create the time axis
    times = np.arange(len(trans))*(1/freq)
    times = times[times <= end_t]
    end_n = len(times)
    
    #rotate the accelerations
    acc_rot = rotate_to_zero(acc, ypr)
    acc_smooth = np.apply_along_axis(lambda x: gaussian_filter(x,2), axis=0, arr =acc[:end_n])
    acc_rot_smooth = np.apply_along_axis(lambda x: gaussian_filter(x,2), axis=0, arr =acc_rot[:end_n])
    
    f, ax = plt.subplots(nrows=1,ncols=2)
    
    
    feature_vector = acc_smooth[:end_n,:]
    ax[0].plot(times,feature_vector*acc_mult)
    ax[0].legend(['a_x', 'a_y', 'a_z','norm'])
    ax[0].set_xlabel('time (s)')
    ax[0].set_ylabel('acceleration (m/s^2)')
    ax[0].set_title('un-adjusted')
    
    feature_vector = acc_rot_smooth[:end_n,:].reshape(-1,1)
    rotPCAData, new_coords = PCA_rotate_data(feature_vector, end_n,nonlinear=nonlinear)
    rotPCAData = np.array(rotPCAData).reshape(end_n,-1)
    new_coords = np.array(new_coords)
    ax[1].plot(times,rotPCAData*acc_mult)
    ax[1].legend(['a_x', 'a_y', 'a_z','norm'])
    ax[1].set_xlabel('time (s)')
    ax[1].set_ylabel('acceleration (m/s^2)')
    ax[1].set_title('adjusted')
    
    f.set_size_inches((10,5))
    
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(acc_rot_smooth[:,0]*acc_mult, acc_rot_smooth[:,1]*acc_mult, acc_rot_smooth[:,2]*acc_mult, c=np.arange(acc_rot_smooth.shape[0]),cmap='jet',s=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    mean_a = np.mean(acc_rot_smooth, axis=0)*acc_mult
    ax.quiver(mean_a[0],mean_a[1],mean_a[2],new_coords[0,0],new_coords[0,1],new_coords[0,2], color = 'r')
    ax.quiver(mean_a[0],mean_a[1],mean_a[2],new_coords[1,0],new_coords[1,1],new_coords[1,2], color = 'b')
    ax.quiver(mean_a[0],mean_a[1],mean_a[2],new_coords[2,0],new_coords[2,1],new_coords[2,2], color = 'g')
    ax.set_title('rotated')
    
    return 


    
    



plot_adjusted_vs_unadjusted(acc_ms, ypr_ms, 50, 2.56,acc_mult=1)


times = np.arange(len(std_ms))*0.02

plt.figure()
plt.plot(times,std_ms.pitch)
plt.plot(times,std_ms.roll)
plt.plot(times,std_ms.yaw)
plt.ylim((-np.pi,np.pi))
plt.xlabel('time (s)')
plt.ylabel('angle (rad)')
plt.legend(['pitch','roll','yaw'])
plt.savefig('motionsense_sitting')

plt.figure()
plt.plot(times,std_ms.rot_x)
plt.plot(times,std_ms.rot_y)
plt.plot(times,std_ms.rot_z)

plt.figure()
plt.plot(times,np.cumsum(std_ms.rot_x))
plt.plot(times,np.cumsum(std_ms.rot_y))
plt.plot(times,np.cumsum(std_ms.rot_z))

# trans = scipy.spatial.transform.Rotation.from_euler('zxy',std_ms[['yaw', 'pitch', 'roll']]).as_matrix()
std_ma = np.apply_along_axis(lambda x:np.deg2rad(x),arr = std_4[['azimuth', 'pitch', 'roll']], axis=1)
trans = scipy.spatial.transform.Rotation.from_euler('zxy',std_ma).as_matrix()


pts = []

for i in range(len(std_ms)):
    pts.append(trans[i] @ np.array([1,0,0]).reshape(3,1))

pts = np.array(pts)

   
# fig = plt.figure()
ax = plt.axes(projection='3d')
u, v = np.mgrid[0:2*np.pi:200j, 0:np.pi:100j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
ax.plot_surface(x, y, z, color="k", alpha = .1)

ax.scatter3D(pts[:,0], pts[:,1], pts[:,2], c=np.arange(len(pts)),cmap='jet',s=0.5)


scipy.spatial.transform.Rotation.from_euler('zxy',std_ma).as_quat()
def rotate_quat(p,q):
    p_new = (q*p)*q.conjugate
    return p_new


for user in range(1,62):
    #user=2
    user = str(user)
    try:
        jog = pd.read_csv('/Volumes/GoogleDrive/My Drive/Harvey Mudd/Work/Summer 2020/project_data/MobiAct_Dataset_v2.0/Annotated Data/JOG/JOG_'+user+'_1_annotated.csv')
        jog = jog.iloc[500:]
        ypr_ma = np.apply_along_axis(lambda x:np.deg2rad(x),arr = jog[['azimuth', 'pitch', 'roll']], axis=1)
        acc_ma = jog[['acc_x', 'acc_y', 'acc_z']].to_numpy()
        plot_adjusted_vs_unadjusted(acc_ma, ypr_ma, 87, 2.56,acc_mult=1/9.8)
        plt.title(user)
    except:
        continue




plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(acc_rot_smooth[:,0], acc_rot_smooth[:,1], acc_rot_smooth[:,2], c=np.arange(acc_rot_smooth.shape[0]),cmap='jet',s=0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('rotated')


plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(acc_smooth[:,0], acc_smooth[:,1], acc_smooth[:,2], c=np.arange(acc_smooth.shape[0]),cmap='jet',s=0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.title('not rotated')


plot_adjusted_vs_unadjusted(acc_ms, rpy_ms, 87, 2.56,acc_mult=9.8)




plt.figure()
ax = plt.axes(projection='3d')
vel = np.cumsum(acc_rot_smooth,axis=1)
ax.scatter3D(vel[:,0], vel[:,1], vel[:,2], c=np.arange(acc_rot.shape[0]),cmap='jet',s=0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.figure()
plt.plot(vel)