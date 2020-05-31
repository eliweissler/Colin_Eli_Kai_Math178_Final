from pyquaternion import Quaternion

import pandas as pd
import numpy as np
from numpy import random
from numpy.linalg import norm

def rotate_data(feature_vector, align_vector):
    """
    We want to rotate our data in some frame
    to a new frame by the same magnitude.

    best way to do this is using quaternions
    from a memory POV i think
    """
    rotated_data = []

    for i in range(0, len(feature_vector), 3):
        # the data we are rotating
        u = data[i:i+3]

        # unit vector to align to (e.g. gravity)
        v = align_vector

        # angle between vectors
        theta = np.arccos(np.dot(u,v)/(norm(u)*norm(v)))

        # find axis perpendicular to u and v (cross product) to rotate about
        r = np.cross(u,v)
        # rotation axis must be a unit vector
        r = r/norm(r)

        # quaternion to rotate [0,ux,uy,uz]
        p = Quaternion(imaginary = u_test)

        # quaternion rotation matrix, rotates angle theta about axis r
        q = Quaternion(axis=r, angle = theta)
        q_u_new = (q*p)*q.conjugate

        rotated_data += [q_u_new.imaginary] #take the vector part

    return rotated_data