import pandas as pd
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
import scipy
from sklearn.decomposition import PCA

# def spline_accelerometer(feature_vec):
#     """
#     Given a 128 segment accelerometer feature vector separated
#     as [a_x1, a_y1, a_z1, a_x2, ... , a_z128]

#     we fit the data with a univariate spline
#     """

#     xline = feature_vec[0::3]
#     yline = feature_vec[1::3]
#     zline = feature_vec[2::3]


#     #spline the accleration, aka alpha'
#     num = len(feature_vec)/3
#     t = np.arange(0,num) #define time interval
#     a_x_spline = IUS(t, xline)
#     a_y_spline = IUS(t, yline)
#     a_z_spline = IUS(t, zline)

#     return a_x_spline, a_y_spline, a_z_spline



# def calc_torsion(feature_vec):
#     """
#     given accelerometer feature vector (which we call alpha prime),
#     calculate the torsion at each 0.02 second time interval

#     by finding first and second derivatives of accelerometer data
#     """
#     #get alpha'
#     a_x_spline, a_y_spline, a_z_spline = spline_accelerometer(feature_vec)

#     #find alpha''
#     ap_x_spline = a_x_spline.derivative()
#     ap_y_spline = a_y_spline.derivative()
#     ap_z_spline = a_z_spline.derivative()

#     #find alpha'''
#     app_x_spline = ap_x_spline.derivative()
#     app_y_spline = ap_y_spline.derivative()
#     app_z_spline = ap_z_spline.derivative()

#     #define time interval
#     num = len(feature_vec)/3
#     t = np.arange(0,num) #define time interval

#     #calculate torsion
#     torL = []
#     for tt in t:
#         #calculate vector
#         v_1 = np.array([a_x_spline(tt), a_y_spline(tt), a_z_spline(tt)])
#         v_2 = np.array([ap_x_spline(tt), ap_y_spline(tt), ap_z_spline(tt)])
#         v_3 = np.array([app_x_spline(tt), app_y_spline(tt), app_z_spline(tt)])

#         #calculate the torsion and shove it in list
#         torsion = 3 * np.linalg.det([v_1, v_2, v_3])  \
#                    /np.linalg.norm(np.cross(v_1, v_2))**2
#         torL += [torsion]
#     return torL



# def calc_curvature(feature_vec):
#     """
#     given accelerometer feature vector (which we call alpha prime),
#     calculate the curvature at each 0.02 second time interval

#     by finding first and second derivatives of accelerometer data
#     """
#     #get alpha'
#     a_x_spline, a_y_spline, a_z_spline = spline_accelerometer(feature_vec)

#     #find alpha''
#     ap_x_spline = a_x_spline.derivative()
#     ap_y_spline = a_y_spline.derivative()
#     ap_z_spline = a_z_spline.derivative()

#     #define time interval
#     num = len(feature_vec)/3
#     t = np.arange(0,num) #define time interval
#     #calculate curvature
#     curvL = []
#     for tt in t:
#         #calculate vector
#         v_1 = np.array([a_x_spline(tt), a_y_spline(tt), a_z_spline(tt)])
#         v_2 = np.array([ap_x_spline(tt), ap_y_spline(tt), ap_z_spline(tt)])

#         #calculate curvature
#         curvature = 2 * np.linalg.norm(np.cross(v_1, v_2)) \
#                     /np.linalg.norm(v_1)**(3/2)
#         curvL += [curvature]

#     return curvL

def spline_accelerometer(feature_vec, kk = 4):
    """
    Given a 128 segment accelerometer feature vector separated
    as [a_x1, a_y1, a_z1, a_x2, ... , a_z128]

    we fit the data with a univariate spline
    """

    xline = feature_vec[0::3]#*np.hanning(len(feature_vec)//3)
#     xline = np.real(np.fft.ifft(np.pad(np.fft.fft(xline)[103:-103],103)))
    
    yline = feature_vec[1::3]#*np.hanning(len(feature_vec)//3)
#     yline = np.real(np.fft.ifft(np.pad(np.fft.fft(yline)[103:-103],103)))

    zline = feature_vec[2::3]#*np.hanning(len(feature_vec)//3)
#     zline = np.real(np.fft.ifft(np.pad(np.fft.fft(zline)[103:-103],103)))

    #spline the accleration, aka alpha'
    num = len(feature_vec)/3
    t = np.arange(0,num) #define time interval
    a_x_spline = IUS(t, xline, k = kk)
    a_y_spline = IUS(t, yline, k = kk)
    a_z_spline = IUS(t, zline, k = kk)

    return a_x_spline, a_y_spline, a_z_spline


def calc_torsion(feature_vec):
    """
    given accelerometer feature vector (which we call alpha prime),
    calculate the torsion at each 0.02 second time interval

    by finding first and second derivatives of accelerometer data
    """
    #get alpha
    a_x_spline, a_y_spline, a_z_spline = spline_accelerometer(feature_vec)

    #find alpha'
    ap_x_spline = a_x_spline.derivative()
    ap_y_spline = a_y_spline.derivative()
    ap_z_spline = a_z_spline.derivative()

    #find alpha''
    app_x_spline = ap_x_spline.derivative()
    app_y_spline = ap_y_spline.derivative()
    app_z_spline = ap_z_spline.derivative()
    
    #find alpha'''
    appp_x_spline = app_x_spline.derivative()
    appp_y_spline = app_y_spline.derivative()
    appp_z_spline = app_z_spline.derivative()

    #define time interval
    num = len(feature_vec)/3
    t = np.arange(0,num) #define time interval

    #calculate torsion
    torL = []
    for tt in t:
        #calculate vector
        v_1 = np.array([ap_x_spline(tt), ap_y_spline(tt), ap_z_spline(tt)])
        v_2 = np.array([app_x_spline(tt), app_y_spline(tt), app_z_spline(tt)])
        v_3 = np.array([appp_x_spline(tt), appp_y_spline(tt), appp_z_spline(tt)])

        #calculate the torsion and shove it in list
        num = np.linalg.det([v_1, v_2, v_3])  
        denom = np.linalg.norm(np.cross(v_1, v_2))**2
        torsion = num/denom
#         print(denom)
        torL += [torsion]
    return torL



def calc_curvature(feature_vec):
    """
    given accelerometer feature vector (which we call alpha prime),
    calculate the curvature at each 0.02 second time interval

    by finding first and second derivatives of accelerometer data
    """
    #get alpha
    a_x_spline, a_y_spline, a_z_spline = spline_accelerometer(feature_vec)

    #find alpha'
    ap_x_spline = a_x_spline.derivative()
    ap_y_spline = a_y_spline.derivative()
    ap_z_spline = a_z_spline.derivative()

    #find alpha''
    app_x_spline = ap_x_spline.derivative()
    app_y_spline = ap_y_spline.derivative()
    app_z_spline = ap_z_spline.derivative()
    
    #find alpha'''
    appp_x_spline = app_x_spline.derivative()
    appp_y_spline = app_y_spline.derivative()
    appp_z_spline = app_z_spline.derivative()

    #define time interval
    num = len(feature_vec)/3
    t = np.arange(0,num) #define time interval
    #calculate curvature
    curvL = []
    for tt in t:
        #calculate vector
        v_1 = np.array([ap_x_spline(tt), ap_y_spline(tt), ap_z_spline(tt)])
        v_2 = np.array([app_x_spline(tt), app_y_spline(tt), app_z_spline(tt)])

        #calculate curvature
        num = np.linalg.norm(np.cross(v_1, v_2)) 
        denom = np.linalg.norm(v_1)**3
        curvature = num/denom
#         print(denom)
        curvL += [curvature]

    return curvL



def calc_fft(feature_vec, smooth = False):
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
    accXYZ = feature_vec.reshape(len(feature_vec),3)
    pca = PCA(n_components = 3)

    pca.fit(accXYZ)
    eigVals = pca.explained_variance_
    eigVects = pca.components_

    return eigVals, eigVects

def calc_manifold_feats(feature_vec, smooth = False):
    """

    Parameters
    ----------
    feature_vec : TYPE: pandas df row
        computes the curvature, curvature fft, torsion, and torsion fft on the given 
        df row.

    Returns
    -------
    df row.
    
    """
    # calc curv
    curv = calc_curvature(feature_vec)
    # calc torsion
    tors = calc_torsion(feature_vec)
    if smooth:
        curv = scipy.ndimage.filters.gaussian_filter1d(curv,2)
        tors = scipy.ndimage.filters.gaussian_filter1d(tors,2)

    # do FFT magnitude, only need to save the first half, since fft is symmetric
    curvFFT = np.abs(np.fft.fft(curv))[:int(len(curv)/2)]
    torsFFT = np.abs(np.fft.fft(tors))[:int(len(tors)/2)]

    # new row will be ordered as the curvature, torsion, curv fft, tors fft
    row = np.concatenate([curvFFT,torsFFT])
    
    return row
                   
def hand_crafted(feature_vec):
    """
    Assumes we are given a df with only acceleration

    Parameters
    ----------
    feature_vec : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    ax = feature_vec[0::3]
    ay = feature_vec[1::3]
    az = feature_vec[2::3]
    # average acceleration 
    avgAcc = [ax.mean(),ay.mean(),az.mean()]
    # std of acceleration 
    stdAcc = [ax.std(),ay.std(),az.std()]
    
    # do magnitude calculations
    # create array of n x [ax,ay,az] for magnitude of the vector
    arr = np.array([ax,ay,az]).T
    n = arr.shape[0]
    magnitude = np.linalg.norm(arr,axis =1)
    
    # avgMag 
    avgMag = magnitude.sum()/n
    stdMag = magnitude.std()
    
    # do absolute mean of acceleration for each
    absAvgAcc = [ax.abs().mean(),ay.abs().mean(),az.abs().mean()]
    
    return np.array(avgAcc + stdAcc + absAvgAcc + [avgMag,stdMag])
    
def all_feats(feature_vec, smooth = False):
    """
    calculate all of our features

    Parameters
    ----------
    feature_vec : TYPE
        DESCRIPTION.
    smooth : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    hc = hand_crafted(feature_vec)
    manifold = calc_manifold_feats(feature_vec)
    
    return np.append(manifold,hc)
                   
                   