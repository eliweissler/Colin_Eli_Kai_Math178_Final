import pandas as pd
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
import scipy
from sklearn.decomposition import PCA 

def spline_accelerometer(feature_vec):
    """
    Given a 128 segment accelerometer feature vector separated
    as [a_x1, a_y1, a_z1, a_x2, ... , a_z128]

    we fit the data with a univariate spline
    """

    xline = feature_vec[0::3]
    yline = feature_vec[1::3]
    zline = feature_vec[2::3]


    #spline the accleration, aka alpha'
    t = np.linspace(0, 2.54, 128) #define time interval
    a_x_spline = IUS(t, xline)
    a_y_spline = IUS(t, yline)
    a_z_spline = IUS(t, zline)

    return a_x_spline, a_y_spline, a_z_spline



def calc_torsion(feature_vec):
    """
    given accelerometer feature vector (which we call alpha prime), 
    calculate the torsion at each 0.02 second time interval

    by finding first and second derivatives of accelerometer data
    """
    #get alpha'
    a_x_spline, a_y_spline, a_z_spline = spline_accelerometer(feature_vec)
    
    #find alpha''
    ap_x_spline = a_x_spline.derivative()
    ap_y_spline = a_y_spline.derivative()
    ap_z_spline = a_z_spline.derivative()

    #find alpha'''
    app_x_spline = ap_x_spline.derivative()
    app_y_spline = ap_y_spline.derivative()
    app_z_spline = ap_z_spline.derivative()

    #define time interval
    t = np.linspace(0, 2.54, 128) 

    #calculate torsion
    torL = []
    for tt in t:
        #calculate vector
        v_1 = np.array([a_x_spline(tt), a_y_spline(tt), a_z_spline(tt)])
        v_2 = np.array([ap_x_spline(tt), ap_y_spline(tt), ap_z_spline(tt)])
        v_3 = np.array([app_x_spline(tt), app_y_spline(tt), app_z_spline(tt)])
        
        #calculate the torsion and shove it in list
        torsion = 3 * np.linalg.det([v_1, v_2, v_3])  \
                   /np.linalg.norm(np.cross(v_1, v_2))
        torL += [torsion]
    return torL



def calc_curvature(feature_vec):
    """
    given accelerometer feature vector (which we call alpha prime), 
    calculate the curvature at each 0.02 second time interval

    by finding first and second derivatives of accelerometer data
    """
    #get alpha'
    a_x_spline, a_y_spline, a_z_spline = spline_accelerometer(feature_vec)
    
    #find alpha''
    ap_x_spline = a_x_spline.derivative()
    ap_y_spline = a_y_spline.derivative()
    ap_z_spline = a_z_spline.derivative()

    #define time interval
    t = np.linspace(0, 2.54, 128) 

    #calculate curvature
    curvL = []
    for tt in t:
        #calculate vector
        v_1 = np.array([a_x_spline(tt), a_y_spline(tt), a_z_spline(tt)])
        v_2 = np.array([ap_x_spline(tt), ap_y_spline(tt), ap_z_spline(tt)])
        
        #calculate curvature
        curvature = 2 * np.linalg.norm(np.cross(v_1, v_2)) \
                    /np.linalg.norm(v_1)**(3/2)
        curvL += [curvature]

    return curvL

def calc_fft(feature_vec, smooth = True):
    """
    takes in feature vector, splits it into its 3 components, and does the fft of each
    returns magnitude of fft for each component 
    """
    # separate components
    xline = feature_vec[0::3]
    yline = feature_vec[1::3]
    zline = feature_vec[2::3]
    
    # smooth the data, with sigma =2 
    if smooth:
        xline = scipy.ndimage.filters.gaussian_filter1d(xline,2)
        yline = scipy.ndimage.filters.gaussian_filter1d(yline,2)
        zline = scipy.ndimage.filters.gaussian_filter1d(zline,2)

    
    # do fft of each
    xFFT = np.abs(np.fft.fftshift(np.fft.fft(xline)))
    yFFT = np.abs(np.fft.fftshift(np.fft.fft(yline)))
    zFFT = np.abs(np.fft.fftshift(np.fft.fft(zline)))

    return xFFT, yFFT, zFFT

def calc_pca(feature_vec):
    """
    takes in feature vector, returns eigenvectors and eigenvalues based on pca
    """
    # reshape the data
    accXYZ = feature_vec.reshape(128,3)
    pca = PCA(n_components = 3)
        
    pca.fit(accXYZ)
    eigVals = pca.explained_variance_
    eigVects = pca.components_

    return eigVals, eigVects
        
    
    