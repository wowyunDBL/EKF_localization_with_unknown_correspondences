'''math tool'''
import csv
import math
import numpy as np
from numpy import cos, sin, arctan2
from scipy.spatial import distance as dist
from shapely import geometry

'''plot tool'''
import matplotlib.pyplot as plt
from matplotlib import animation

'''image tool'''
import cv2
import statistics as sta

'''mat loading'''
from scipy.io import loadmat

import sys

def plot_traj(true_states, belief_states, markers):
    x_tr, y_tr, th_tr = true_states
    x_guess, y_guess = belief_states

    radius = 0.5
    
    world_bounds = [-10,10]
    fig, ax = plt.subplots(figsize=(10,10),dpi=120)
    ax = plt.axes(xlim=world_bounds, ylim=world_bounds)
    ax.set_aspect('equal')

    '''plot landmarkers'''
    plt.scatter(markers[0], markers[1], marker='X',s=100, color='g', label='ref landmarks')
    number_of_point=12
    piece_rad = np.pi/(number_of_point/2)
    neg_bd = []
    for i in range(number_of_point):
        neg_bd.append((markers[0]+markers[2]*np.cos(piece_rad*i), markers[1]+markers[2]*np.sin(piece_rad*i)))
    plt.scatter(neg_bd[:,0], neg_bd[:,1], c='k', s=10)

    '''plot traj'''
    plt.scatter(x_tr[0], y_tr[0], color='b', label="Actual", s=10)
    plt.scatter(x_guess[0], y_guess[0], color='r', label="Predicted", s=10)

    '''plot final state'''
    plt.scatter(x_tr[0][x_tr.shape[1]-1],y_tr[0][y_tr.shape[1]-1], s=500, color='y', ec='k')
    plt.plot( [x_tr[0][x_tr.shape[1]-1], x_tr[0][x_tr.shape[1]-1] + radius*cos(th_tr[0][x_tr.shape[1]-1])], 
               [y_tr[0][x_tr.shape[1]-1], y_tr[0][x_tr.shape[1]-1] + radius*sin(th_tr[0][x_tr.shape[1]-1])], color='k' )

    plt.legend()
    plt.show()

def get_mu_bar(prev_mu, velocity, omega, angle, dt):
    ratio = velocity/omega
    m = np.array([[(-ratio*sin(angle))+(ratio*sin(angle+omega*dt))],
                  [(ratio*cos(angle))-(ratio*cos(angle+omega*dt))],
                  [omega*dt]])
    return prev_mu + m

def get_observed_lm(mu_bar, global_lm):

    return obs_lm_x, obs_lm_y, obs_lm_radi

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

if __name__ == "__main__":
    dt = .1
    t = np.arange(0, 20+dt, dt)
    t = np.reshape(t, (1,-1))
    
    ''' belief (estimate from EKF) '''
    mu_x = np.zeros(t.shape)
    mu_y = np.zeros(t.shape)
    mu_theta = np.zeros(t.shape)   # radians
    ''' starting belief - initial condition (robot pose) '''
    mu_x[0,0] = -5
    mu_y[0,0] = -3
    mu_theta[0,0] = 90/180*np.pi
    '''initial uncertainty in the belief'''
    sigma = np.array([[1, 0, 0],  # x
                      [0, 1, 0],  # y
                      [0, 0, .1] ]) # theta
    
    '''command velocity'''
    v_c = 1 + 0.5*cos(2*np.pi*(0.2)*t)
    omg_c = -0.2 + 2*cos(2*np.pi*(0.6)*t)
    '''noise in the command velocities (translational and rotational)'''
    alpha = np.array([.1, .01, .01, .1])
    alpha_1, alpha_2, alpha_3, alpha_4 = alpha

    '''landmarks'''
    lm_x = [6,-7,6]
    lm_y = [4,8,-4]
    lm_radi = [0.2, 0.5, 0.3]
    assert (len(lm_x)==len(lm_y))
    '''std deviation of range and bearing sensor noise for each landmark'''
    std_dev_x = .1
    std_dev_y = .05
    std_dev_radi = .5
    '''uncertainty due to measurement noise'''
    Q_t = np.array([ [std_dev_x, 0, 0],
                     [0, std_dev_y, 0],
                     [0, 0, std_dev_radi] ] )

    '''ground truth'''
    x_pos_true = np.zeros(t.shape)
    y_pos_true = np.zeros(t.shape)
    theta_pos_true = np.zeros(t.shape)
    '''set ground truth data by calculation'''
    x_pos_true[0,0] = -5
    y_pos_true[0,0] = -3
    theta_pos_true[0,0] = 0.5*np.pi
    velocity = v_c + np.random.normal( scale=np.sqrt( (alpha_1*(v_c**2)) + (alpha_2*(omg_c**2)) ))
    omega = omg_c + np.random.normal( scale=np.sqrt( (alpha_3*(v_c**2)) + (alpha_4*(omg_c**2)) ))
    for timestep in range(1, t.size):
        prev_state = np.array([x_pos_true[0, timestep-1],
                               y_pos_true[0, timestep-1],
                               theta_pos_true[0, timestep-1]])
        prev_state = np.reshape(prev_state,(-1,1))
        theta_prev = theta_pos_true[0 , timestep-1]
        next_state = get_mu_bar(prev_state, velocity[0, timestep], 
                                    omega[0, timestep], theta_prev, dt)
        x_pos_true[0, timestep] = next_state[0]
        y_pos_true[0, timestep] = next_state[1]
        theta_pos_true[0, timestep] = next_state[2]
    # plot_traj((x_pos_true, y_pos_true, theta_pos_true), (x_pos_true, y_pos_true), (lm_x, lm_y))
    
    '''run KF'''
    # mu = np.array([[mu_x[0,0]],[mu_y[0,0]],[mu_theta[0,0]]])
    for i in range(1, t.size):
        print(">>>>new ietration")
        
        # This is only for temperary covariance
        curr_v = v_c[0,i]
        curr_w = omg_c[0,i]
        prev_theta = mu_theta[0,i-1]

        G_t = get_G_t(curr_v, curr_w, prev_theta, dt)
        V_t = get_V_t(curr_v, curr_w, prev_theta, dt)
        M_t = get_M_t(alpha, curr_v, curr_w)

        '''prediction step'''
        # mu_bar = get_mu_bar(mu, curr_v, curr_w, prev_theta, dt)
        mu = np.array([ [x_pos_true[0,i]],[y_pos_true[0,i]],[theta_pos_true[0,i]] ])
        mu_bar += make_noise(Q_t)
        sigma_bar = (G_t @ sigma @ (G_t.T)) + (V_t @ M_t @ (V_t.T))

        '''correction (updating belief based on landmark readings)'''
        bel_x = mu_bar[0,0]
        bel_y = mu_bar[1,0]
        bel_theta = mu_bar[2,0]
        # cannot observe all lm !
        obs_lm_x, obs_lm_y, obs_lm_radi = get_observed_lm(mu, (lm_x, lm_y, lm_radi))
        for k in range(len(obs_lm_y)):
            npLikelihood = np.array([])
            list_z_hat = []
            list_S_t = []
            list_H_t = []
            
            obs_k_x = obs_lm_x[k]
            obs_k_y = obs_lm_y[k]
            obs_k_radi = obs_lm_radi[k]
            '''get the sensor measurement'''
            # real_x = x_pos_true[0,i]
            # real_y = y_pos_true[0,i]
            # real_tehta = theta_pos_true[0,i]
            # diff_x = obs_k_x - real_x
            # diff_y = obs_k_y - real_y
            # q = (diff_x ** 2) + (diff_y ** 2)
            z_true = np.array([ [obs_k_x],
                                [obs_k_y],
                                [obs_k_radi] ])
            z_true += make_noise(Q_t)
            for j in range(len(lm_y)):
                m_j_x = lm_x[j]
                m_j_y = lm_y[j]
                m_j_radi = lm_radi[j]
                                              
                '''likelihood'''
                diff_x = m_j_x - bel_x
                diff_y = m_j_y - bel_y
                # q = (diff_x ** 2) + (diff_y ** 2)
                z_hat = np.array([ [diff_x],
                                   [diff_y],
                                   [obs_k_radi] ])

                H_t = np.array([ [-diff_x / np.sqrt(q), -diff_y / np.sqrt(q), 0],
                                 [diff_y / q, -diff_x / q, -1],
                                 [0, 0, 1] ])
                S_t = (H_t @ sigma_bar @ (H_t.T)) + Q_t
                likelihood = np.sqrt(np.linalg.det(2*np.pi*S_t)) * math.exp(-0.5*((z_true-z_hat).T)@(np.linalg.inv(S_t))@(z_true-z_hat))
                npLikelihood = np.append(npLikelihood,likelihood)
                
                list_z_hat.append(z_hat)
                list_S_t.append(S_t)
                list_H_t.append(H_t)
            '''maximum likelihood'''
            maxLikelihood = npLikelihood.argmax()
            H_t = list_H_t.pop(maxLikelihood)
            S_t = list_S_t.pop(maxLikelihood)
            z_hat = list_z_hat.pop(maxLikelihood)
            '''kalman gain and update belief'''
            K_t = sigma_bar @ (H_t.T) @ np.linalg.inv(S_t)
            mu_bar = mu_bar+K_t@(z_true-z_hat)
            sigma_bar = (np.identity(sigma_bar.shape[0])-(K_t @ H_t)) @ sigma_bar
        '''update belief'''
        mu = mu_bar
        sigma = sigma_bar
        mu_x[0 , i] = mu[0 , 0]
        mu_y[0 , i] = mu[1 , 0]
        mu_theta[0 , i] = mu[2 , 0]
    plot_traj((x_pos_true, y_pos_true, theta_pos_true), (mu_x, mu_y), (lm_x, lm_y, lm_radi)) 