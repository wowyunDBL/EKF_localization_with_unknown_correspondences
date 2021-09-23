#!usr/bin/env python3
'''math tool'''
import csv
import math
import numpy as np
from numpy import cos, sin, arctan2
from scipy.spatial import distance as dist
#from shapely import geometry

'''plot tool'''
import matplotlib.pyplot as plt
from matplotlib import animation

'''image tool'''
import cv2
import statistics as sta

'''mat loading'''
from scipy.io import loadmat

import sys

def get_mu_bar(prev_mu, velocity, omega, angle, dt):
    ratio = velocity/omega
    m = np.array([[(-ratio*sin(angle))+(ratio*sin(angle+omega*dt))],
                  [(ratio*cos(angle))-(ratio*cos(angle+omega*dt))],
                  [omega*dt]])
    return prev_mu + m

def get_observed_lm(index):
    file_path = "/home/ncslaber/110-1/210922_EKF-fusion-test/zigzag_bag/"
    # obs_lm_x = []
    # obs_lm_y = []
    # obs_lm_radi = []
    np_z_true = []
    cX_utm_loc = np.load(file_path+"found_center/"+str(index)+'-x.npy')
    cY_utm_loc = np.load(file_path+"found_center/"+str(index)+'-y.npy')
    obs_lm_x = np.load(file_path+"found_center/"+str(index)+'-q_x.npy')
    obs_lm_y = np.load(file_path+"found_center/"+str(index)+'-q_y.npy')
    obs_lm_radi = np.load(file_path+"found_center/"+str(index)+'-r.npy')
    obs_lm_radi = obs_lm_radi/ 10
    for c in range( len(obs_lm_x) ):
        diff_x = obs_lm_x[c] 
        diff_y = obs_lm_y[c] 
        diff_theta = arctan2(diff_y, diff_x)

        q = (diff_x ** 2) + (diff_y ** 2)
        z_true = np.array([ [np.sqrt(q)],
                            [diff_theta],
                            [obs_lm_radi[c]] ])

        np_z_true=np.append(np_z_true, z_true)
                
    return np_z_true, cX_utm_loc, cY_utm_loc, obs_lm_radi

def get_G_t(v, w, angle, dt):
    return np.array([
                    [1, 0, ( (-v/w)*cos(angle) ) + ( (v/w)*cos(angle + (w*dt)) ) ],
                    [0, 1, ( (-v/w)*sin(angle) ) + ( (v/w)*sin(angle + (w*dt)) ) ],
                    [0, 0, 1]
                    ])

def get_V_t(v, w, angle, dt):
    v_0_0 = ( -sin(angle) + sin(angle + (w*dt)) ) / w
    v_0_1 = ( (v * (sin(angle) - sin(angle + (w*dt)))) / (w*w) ) + \
        ( (v * cos(angle + (w*dt)) * dt) / w )
    v_1_0 = ( cos(angle) - cos(angle + (w*dt)) ) / w
    v_1_1 = ( -(v * (cos(angle) - cos(angle + (w*dt)))) / (w*w) ) + \
        ( (v * sin(angle + (w*dt)) * dt) / w )
    return np.array([
                    [v_0_0, v_0_1],
                    [v_1_0, v_1_1],
                    [0, dt]
                    ])

def get_M_t(alpha, v, w):
    a_1, a_2, a_3, a_4=alpha
    return np.array([ [( (a_1 * v*v) + (a_2 * w*w) ), 0],
                      [0, ( (a_3 * v*v) + (a_4 * w*w) )] ])

def make_noise(cov_matrix):
    noisy_transition = \
        np.random.multivariate_normal(np.zeros(cov_matrix.shape[0]), cov_matrix)
    return np.reshape(noisy_transition, (-1,1))

if __name__ == '__main__':
