from pyquaternion import Quaternion

import pandas as pd
import numpy as np
from numpy import random
from numpy.linalg import norm
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R

from sklearn.decomposition import KernelPCA

def rotate_data_bad(feature_vector, align_vector):
    """
    We want to rotate our data in some frame
    to a new frame by the same magnitude.

    best way to do this is using quaternions
    from a memory POV i think
    """
    rotated_data = []
    
    print("This probably isn't the function you want.")

    for i in range(0, len(feature_vector), 3):
        # the data we are rotating
        u = feature_vector[i:i+3]

        # unit vector to align to (e.g. gravity)
        v = align_vector

        # angle between vectors
        theta = np.arccos(np.dot(u,v)/(norm(u)*norm(v)))

        # find axis perpendicular to u and v (cross product) to rotate about
        r = np.cross(u,v)
        # rotation axis must be a unit vector
        r = r/norm(r)

        # quaternion to rotate [0,ux,uy,uz]
        p = Quaternion(imaginary = u)

        # quaternion rotation matrix, rotates angle theta about axis r
        q = Quaternion(axis=r, angle = theta)
        q_u_new = (q*p)*q.conjugate

        rotated_data += [q_u_new.imaginary] #take the vector part

    return rotated_data




def PCA_rotate_data(feature_vector, n_points=128, nonlinear=True):
    """
    Rotate the data to align with the principal components
    of our acceleration data.

    Component with largest eigenvector will be z-axis
    Second largest will be y-axis, I guess.
    """

    accXYZ = feature_vector.reshape(n_points,-1)
    if nonlinear:
        pca = KernelPCA(n_components=3, kernel='polynomial')
    else:
        pca = PCA(n_components = 3)
       
    pca.fit(accXYZ)
    
    # pca.explained_variance_: importance of data on each axis aka their important
    # tells us direction of vector, they are the eigenvalues
    if nonlinear:
        eigVals = pca.lambdas_
        eigVects = pca.alphas_
    else:
        eigVals = pca.explained_variance_
        eigVects = pca.components_

    x_index = np.argmin(eigVals)
    z_index = np.argmax(eigVals)
    y_index = list(set([0,1,2]) - set([x_index, z_index]))[0]

    new_x_hat = eigVects[x_index]/norm(eigVects[x_index])
    new_y_hat = eigVects[y_index]/norm(eigVects[y_index])
    new_z_hat = eigVects[z_index]/norm(eigVects[z_index])

    rotPCAData = []

    for i in range(len(accXYZ)):
        v = accXYZ[i]
        new_x = np.dot(new_x_hat, v )
        new_y = np.dot(new_y_hat, v )
        new_z = np.dot(new_z_hat, v )
        rotPCAData += [new_x, new_y, new_z]

    return rotPCAData, [new_x_hat, new_y_hat, new_z_hat]

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
    
    #apply the appropriate roation for each element
    acc_rot = np.array([rotate_quat(Quaternion(imaginary=acc[i,:]), trans[i].conjugate).imaginary for i in range(acc.shape[0])])
    
    return acc_rot

def rotate_quat(p,q):
    p_new = (q*p)*q.conjugate
    return p_new