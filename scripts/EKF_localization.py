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

def get_mu_bar_odom_modle(prev_mu, u, north_heading=False):
    '''hat means in odom frame'''
    angle = prev_mu[2,0]
    print('angle: ', angle)
    # print("prev_mu: ", prev_mu)
    prev_odom_hat, odom_hat = u
    prev_x_hat, prev_y_hat, prev_t_hat = prev_odom_hat[:,0]
    x_hat, y_hat, t_hat = odom_hat[:,0]
    # print('x_hat, y_hat, t_hat: ', x_hat, y_hat, t_hat)
    # print('prev_x_hat, prev_y_hat, prev_t_hat: ', prev_x_hat, prev_y_hat, prev_t_hat)

    diff_x = x_hat - prev_x_hat 
    diff_y = y_hat - prev_y_hat  

    if north_heading == True:
        delta_rot1 = arctan2(diff_y, diff_x) - prev_t_hat
        delta_trans = np.sqrt((diff_x ** 2) + (diff_y ** 2))
        delta_rot2 = t_hat - prev_t_hat - delta_rot1
        # print("angle+delta_rot1:", angle+delta_rot1)
        m = np.array([[-delta_trans*sin(angle+delta_rot1) ],
                      [ delta_trans*cos(angle+delta_rot1) ],
                      [ delta_rot1 + delta_rot2] ])

    else:
        delta_rot1 = arctan2(diff_y, diff_x) - prev_t_hat
        print('prev_t_hat: ', prev_t_hat)
        print('arctan2(diff_y, diff_x): ', arctan2(diff_y, diff_x))
        delta_trans = np.sqrt((diff_x ** 2) + (diff_y ** 2))
        delta_rot2 = t_hat - prev_t_hat - delta_rot1
        print('t_hat: ', t_hat)

        m = np.array([[ delta_trans*cos(angle+delta_rot1) ],
                      [ delta_trans*sin(angle+delta_rot1) ],
                      [ delta_rot1 + delta_rot2] ])
    print('prev_mu + m: ', prev_mu + m)
    print('m: ', m)
    print('prev_mu: ', prev_mu)

    mu = prev_mu + m
    if mu[2,0]>np.pi:
        mu[ 2,0 ] -= 2*np.pi
    if mu[2,0]<-np.pi:
        mu[ 2,0 ] += 2*np.pi
    return mu


def get_G_t(v, w, angle, dt):
    return np.array([
                    [1, 0, ( (-v/w)*cos(angle) ) + ( (v/w)*cos(angle + (w*dt)) ) ],
                    [0, 1, ( (-v/w)*sin(angle) ) + ( (v/w)*sin(angle + (w*dt)) ) ],
                    [0, 0, 1]
                    ])

def get_G_t_odom(u, angle):
    prev_odom_hat, odom_hat = u
    prev_x_hat, prev_y_hat, prev_t_hat = prev_odom_hat[:,0]
    x_hat, y_hat, t_hat = odom_hat[:,0]

    diff_x = x_hat - prev_x_hat 
    diff_y = y_hat - prev_y_hat

    delta_rot1 = arctan2(diff_y, diff_x) - prev_t_hat  
    delta_trans = np.sqrt((diff_x ** 2) + (diff_y ** 2))

    return np.array([
                    [1, 0, -delta_trans*sin(angle+delta_rot1)],
                    [0, 1, delta_trans*cos(angle+delta_rot1)],
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

def get_predict_lm_measure_and_likelihood(diff_x, diff_y, bel_theta, z_true,  m_j_radi, sigma_bar, Q_t, weight_feature=1):
    
    q = (diff_x ** 2) + (diff_y ** 2)
    # bel_theta += np.pi/2
    # if bel_theta > np.pi:
    #     bel_theta = bel_theta - 2*np.pi
    z_hat = np.array([ [np.sqrt(q)],
                        [arctan2(diff_y, diff_x) - bel_theta],
                        [m_j_radi] ])
    if z_hat[1,0] > np.pi:
        z_hat[1,0] = z_hat[1,0] - 2*np.pi

    H_t = np.array([ [-diff_x / np.sqrt(q), -diff_y / np.sqrt(q), 0],
                        [diff_y / q, -diff_x / q, -1],
                        [0,0,0] ])
    S_t = (H_t @ sigma_bar @ (H_t.T)) + Q_t
    
    print("z_hat: ", z_hat)
    print("z_true: ", z_true)
    print('arctan2(diff_y, diff_x): ', arctan2(diff_y, diff_x))
    print('bel_theta', bel_theta)
    diff_z = z_true-z_hat
    diff_z[2,0] *= weight_feature
    likelihood = np.sqrt(np.linalg.det(2*np.pi*S_t)) * math.exp(-0.5*((z_true-z_hat).T)@(np.linalg.inv(S_t))@(z_true-z_hat))

    return z_hat, H_t, S_t, likelihood 

def get_predict_lm_measure(diff_x, diff_y, bel_theta, m_j_radi, sigma_bar, Q_t):
    
    q = (diff_x ** 2) + (diff_y ** 2)
    z_hat = np.array([ [np.sqrt(q)],
                        [arctan2(diff_y, diff_x) - bel_theta],
                        [m_j_radi] ])
    if z_hat[1,0] > np.pi:
        z_hat[1,0] = z_hat[1,0] - 2*np.pi
    H_t = np.array([ [-diff_x / np.sqrt(q), -diff_y / np.sqrt(q), 0],
                        [diff_y / q, -diff_x / q, -1],
                        [0,0,0] ])
    S_t = (H_t @ sigma_bar @ (H_t.T)) + Q_t
    

    return z_hat, H_t, S_t

if __name__ == '__main__':
    prev_mu = np.array([ [-5],[-3],[ 1.57079633] ])
    u = ( np.array([ [-5],[-3],[ 1.57079633] ]), np.array([ [-5.00185229],[-2.96362823],[ 1.67256138] ]) )
    get_mu_bar_odom_modle(prev_mu, u)
    pass