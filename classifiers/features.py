import pandas as pd
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as IUS


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
    t = np.linspace(0, 2.56, 128) #define time interval
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
    t = np.linspace(0, 2.56, 128) 

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
    t = np.linspace(0, 2.56, 128) 

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