#! /usr/bin/env python3
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

'''self module'''
import load_data_utils
import plot_utils
import EKF_localization
import ICP_correspondences
     

def get_noise_landmark_xy(z_true, robot_pose):
    (real_x, real_y, real_tehta) = robot_pose
    r_x = z_true[0][0] * cos(z_true[1][0])
    r_y = z_true[0][0] * sin(z_true[1][0])
    wr_x =  cos(real_tehta)*r_x - sin(real_tehta)*r_y
    wr_x += real_x
    wr_y = sin(real_tehta)*r_x + cos(real_tehta)*r_y
    wr_y += real_y
    return wr_x, wr_y

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
    omg_c = -0.2 + 2*cos(2*np.pi*(1.2)*t)
    '''noise in the command velocities (translational and rotational)'''
    alpha = np.array([.1, .01, .01, .1])
    alpha_1, alpha_2, alpha_3, alpha_4 = alpha

    '''landmarks'''
    lm_x, lm_y, lm_radi = load_data_utils.get_landmark()
    assert (len(lm_x)==len(lm_y))
    '''std deviation of range and bearing sensor noise for each landmark'''
    std_dev_dist = .05
    std_dev_phi = .05
    std_dev_radi = .05
    '''uncertainty due to measurement noise'''
    Q_t = np.array([ [std_dev_dist, 0, 0],
                     [0, std_dev_phi, 0],
                     [0, 0, std_dev_radi] ] )
    Q_t_tmp = np.array([ [std_dev_dist, 0, 0],
                     [0, std_dev_phi/20, 0],
                     [0, 0, std_dev_radi/20] ] )

    '''ground truth'''
    x_pos_true = np.load('/home/ncslaber/class_material/EKF_localization_with_unknown_correspondences/data_ground_truth/x_pos_true.npy')
    y_pos_true = np.load('/home/ncslaber/class_material/EKF_localization_with_unknown_correspondences/data_ground_truth/y_pos_true.npy')
    theta_pos_true = np.load('/home/ncslaber/class_material/EKF_localization_with_unknown_correspondences/data_ground_truth/theta_pos_true.npy')
    
    '''set ground truth data by calculation'''
    # load_data_utils.generate_ground_truth(t)
    
    '''# of observed landmarks'''
    obs_lm_number = np.zeros(t.shape)
    cols=[]

    '''run EKF'''
    mu = np.array([ [mu_x[0,0]],[mu_y[0,0]],[mu_theta[0,0]] ])
    for i in range(1, t.size):
        print(">>>>new ietration: "+str(i))
        flag = False
        icp_flag = False
        
        # This is only for temperary covariance
        curr_v = v_c[0,i]
        curr_w = omg_c[0,i]
        prev_theta = mu_theta[0,i-1]

        '''prediction step'''
        G_t = EKF_localization.get_G_t(curr_v, curr_w, prev_theta, dt)
        V_t = EKF_localization.get_V_t(curr_v, curr_w, prev_theta, dt)
        M_t = EKF_localization.get_M_t(alpha, curr_v, curr_w)

        mu_bar = EKF_localization.get_mu_bar(mu, curr_v, curr_w, prev_theta, dt)
        # mu = np.array([ [x_pos_true[0,i]],[y_pos_true[0,i]],[theta_pos_true[0,i]] ])
        sigma_bar = (G_t @ sigma @ (G_t.T)) + (V_t @ M_t @ (V_t.T))

        '''correction (updating belief based on landmark readings)'''
        bel_x = mu_bar[0,0]
        bel_y = mu_bar[1,0]
        bel_theta = mu_bar[2,0]

        '''measured landmarks'''
        real_x = x_pos_true[0,i]
        real_y = y_pos_true[0,i]
        real_tehta = theta_pos_true[0,i]
        obs_lm_x, obs_lm_y, obs_lm_radi= load_data_utils.get_observed_lm_for_sim( [real_x,real_y,real_tehta], (lm_x, lm_y, lm_radi) )
        
        '''generate noise measurement'''
        np_z_true=np.array([[],[],[]])
        np_z_hat=np.array([[],[],[]])
        np_obs_x_m_noise=np.array([])
        np_obs_y_m_noise=np.array([])
        for k in range(len(obs_lm_x)):
                       
            obs_k_x = obs_lm_x[k]
            obs_k_y = obs_lm_y[k]
            obs_k_radi = obs_lm_radi[k]
            
            '''get the sensor measurement'''
            diff_x = obs_k_x - real_x
            diff_y = obs_k_y - real_y
            q = (diff_x ** 2) + (diff_y ** 2)
            z_true = np.array([ [np.sqrt(q)],
                                [arctan2(diff_y, diff_x) - real_tehta],
                                [obs_k_radi] ])
            print('z_true:', z_true)
            z_true += EKF_localization.make_noise(Q_t_tmp) 
            print('z_true_noise:', z_true)

            '''save real measured data'''
            np_z_true=np.append(np_z_true, z_true)
            obs_lm_x_noise, obs_lm_y_noise = get_noise_landmark_xy(z_true, (real_x, real_y, real_tehta))
            np_obs_x_m_noise=np.append(np_obs_x_m_noise,obs_lm_x_noise)
            np_obs_y_m_noise=np.append(np_obs_y_m_noise,obs_lm_y_noise)
        
        np_z_true = np.reshape(np_z_true, (-1,3))
        np_z_true = np_z_true.T
        
        ''' find correspondence'''
        if len(obs_lm_x) > 0:
            if len(obs_lm_x) <= 2:
                if len(obs_lm_x) == 1:
                    ''' find correspondence by maximum likelihood '''
                    npLikelihood = np.array([])
                    list_z_hat = []
                    list_S_t = []
                    list_H_t = []
                    print('>>>> find correspondence by maximum likelihood')
                    P = np.vstack((lm_x, lm_y))
                    U = np.vstack((np_obs_x_m_noise, np_obs_y_m_noise))
                    D = dist.cdist(U.T, P.T)
                    print('D: ',D)
                    cols = ICP_correspondences.get_correspondences_in_range(D[0], dist=2)
                    print('radius under 2m: ',cols)

                    for k in range(len(cols)):
                        
                        m_j_x = lm_x[cols[k]]
                        m_j_y = lm_y[cols[k]]
                        m_j_radi = lm_radi[cols[k]]

                        '''get predict measurement'''
                        diff_x = m_j_x - bel_x
                        diff_y = m_j_y - bel_y
                        z_true = np_z_true

                        z_hat, H_t, S_t, likelihood \
                            = EKF_localization.get_predict_lm_measure_and_likelihood( \
                                                diff_x, diff_y, bel_theta, z_true, m_j_radi, sigma_bar, Q_t)
                        
                        npLikelihood = np.append(npLikelihood,likelihood)
                        list_z_hat.append(z_hat)
                        list_S_t.append(S_t)
                        list_H_t.append(H_t)

                    '''maximum likelihood'''
                    maxLikelihood = npLikelihood.argmax()
                    print("matched lm: ", cols[maxLikelihood])
                    print('z_true:', z_true)
                    print('z_hat:', z_hat)
                    print('z_true-z_hat:', z_true-z_hat)
                    print('bel: ',(bel_x, bel_y, bel_theta))
                    
                    H_t = list_H_t.pop(maxLikelihood)
                    S_t = list_S_t.pop(maxLikelihood)
                    z_hat = list_z_hat.pop(maxLikelihood)
                                        
                    '''kalman gain and update belief'''
                    K_t = sigma_bar @ (H_t.T) @ np.linalg.inv(S_t)
                    mu_bar = mu_bar+K_t@(z_true-z_hat)
                    sigma_bar = (np.identity(sigma_bar.shape[0])-(K_t @ H_t)) @ sigma_bar

                    '''save estimated measured data'''
                    np_z_hat=np.append(np_z_hat,z_hat)

                elif len(obs_lm_x) == 2:
                    print('>>>> find correspondence by maximum likelihood')
                    P = np.vstack((lm_x, lm_y))
                    U = np.vstack((np_obs_x_m_noise, np_obs_y_m_noise))
                    D = dist.cdist(U.T, P.T)
                    print('D: ',D)

                    for tmp in [0,1]:
                        ''' find correspondence by maximum likelihood '''
                        npLikelihood = np.array([])
                        list_z_hat = []
                        list_S_t = []
                        list_H_t = []
                        cols = ICP_correspondences.get_correspondences_in_range(D[tmp], dist=2)
                        print('cols: ', cols)
                        for k in range(len(cols)):
                            
                            m_j_x = lm_x[cols[k]]
                            m_j_y = lm_y[cols[k]]
                            m_j_radi = lm_radi[cols[k]]

                            '''get predict measurement'''
                            diff_x = m_j_x - bel_x
                            diff_y = m_j_y - bel_y
                            z_true = np.reshape(np_z_true[:,tmp], (-1,3))
                            z_true = z_true.T

                            z_hat, H_t, S_t, likelihood \
                                = EKF_localization.get_predict_lm_measure_and_likelihood( \
                                                diff_x, diff_y, bel_theta, z_true, m_j_radi, sigma_bar, Q_t, \
                                                weight_feature=2)
                            
                            print('z_true-z_hat:', z_true-z_hat)
                            npLikelihood = np.append(npLikelihood,likelihood)
                            list_z_hat.append(z_hat)
                            list_S_t.append(S_t)
                            list_H_t.append(H_t)

                        '''maximum likelihood'''
                        maxLikelihood = npLikelihood.argmax()
                        print('npLikelihood: ',npLikelihood)
                        print("matched lm: ", cols[maxLikelihood])
                        H_t = list_H_t.pop(maxLikelihood)
                        S_t = list_S_t.pop(maxLikelihood)
                        z_hat = list_z_hat.pop(maxLikelihood)

                        '''save estimated measured data'''
                        np_z_hat=np.append(np_z_hat,z_hat)
                        
                        '''kalman gain and update belief'''
                        K_t = sigma_bar @ (H_t.T) @ np.linalg.inv(S_t)
                        z_true = np.reshape(np_z_true[:,tmp], (-1,3))
                        z_true = z_true.T
                        mu_bar = mu_bar+K_t@(z_true-z_hat)
                        sigma_bar = (np.identity(sigma_bar.shape[0])-(K_t @ H_t)) @ sigma_bar

                        '''save estimated measured data'''
                        np_z_hat=np.append(np_z_hat,z_hat)

            else:
                ''' find correspondence by ICP '''
                print('>>>> find correspondence by ICP')
                icp_flag = True
                P = np.vstack((lm_x, lm_y))
                
                U = np.vstack((np_obs_x_m_noise, np_obs_y_m_noise)) #############(obs_lm_x, obs_lm_y)
                
                cols = ICP_correspondences.get_Rt_by_ICP( P,U)
        
                '''iterative update predict measurement'''
                for k in range(len(obs_lm_x)):
                    m_j_x = lm_x[cols[k]]
                    m_j_y = lm_y[cols[k]]
                    m_j_radi = lm_radi[cols[k]]
                                                    
                    '''get predict measurement'''
                    diff_x = m_j_x - bel_x
                    diff_y = m_j_y - bel_y
                    
                    z_hat, H_t, S_t \
                         = EKF_localization.get_predict_lm_measure(diff_x, diff_y, bel_theta, m_j_radi, sigma_bar, Q_t)
                                        
                    '''kalman gain and update belief'''
                    K_t = sigma_bar @ (H_t.T) @ np.linalg.inv(S_t)
                    print('K: ',K_t)
                    z_true = np.reshape(np_z_true[:,k], (-1,3))
                    z_true = z_true.T
                    print('z_true-z_hat: ',z_true-z_hat)
                    mu_bar = mu_bar+K_t@(z_true-z_hat)
                    sigma_bar = (np.identity(sigma_bar.shape[0])-(K_t @ H_t)) @ sigma_bar

                    '''save estimated measured data'''
                    np_z_hat=np.append(np_z_hat,z_hat)
            np_z_hat = np.reshape(np_z_hat, (-1,3))
            np_z_hat = np_z_hat.T
            # print('np_z_hat: ', np_z_hat)

        '''update belief'''
        mu = mu_bar
        sigma = sigma_bar
        mu_x[0 , i] = mu[0 , 0]
        mu_y[0 , i] = mu[1 , 0]
        mu_theta[0 , i] = mu[2 , 0]
        obs_lm_number[0 , i] = len(np_z_hat)/3
        print("<<<<finish update: "+str(i))

        # if  flag==True: #len(obs_lm_x) != 0:
            # print("difference btw mu_bar and robot_xy_new: ", mu_bar, robot_xy_new)
            # tmp = (mu_bar[:2,0] - robot_xy_new[:,0])
            # plot_utils.plot_traj((x_pos_true, y_pos_true, theta_pos_true), (mu_x, mu_y, mu_theta), (obs_lm_x, obs_lm_y, obs_lm_radi),(lm_x,lm_y,lm_radi),i, \
                        # np_z_hat, np_z_true, (bel_x, bel_y, bel_theta), (real_x, real_y, real_tehta), cols, icp_flag) 
            
        
        if i%10 == 0:
            plot_utils.plot_traj((x_pos_true, y_pos_true, theta_pos_true), (mu_x, mu_y, mu_theta), (obs_lm_x, obs_lm_y, obs_lm_radi),(lm_x,lm_y,lm_radi),i, \
                        np_z_hat, np_z_true, (bel_x, bel_y, bel_theta), (real_x, real_y, real_tehta), cols, icp_flag) 
    
    # np.save('/home/ncslaber/class_material/EKF_localization_with_unknown_correspondences/data_ground_truth/mu_x', mu_x)
    # np.save('/home/ncslaber/class_material/EKF_localization_with_unknown_correspondences/data_ground_truth/mu_y', mu_y)
    # np.save('/home/ncslaber/class_material/EKF_localization_with_unknown_correspondences/data_ground_truth/mu_theta', mu_theta)
    # np.save('/home/ncslaber/class_material/EKF_localization_with_unknown_correspondences/data_ground_truth/obs_lm_number', obs_lm_number)
    # np.save('/home/ncslaber/class_material/EKF_localization_with_unknown_correspondences/data_ground_truth/obs_lm_vector', np_z_true)
    