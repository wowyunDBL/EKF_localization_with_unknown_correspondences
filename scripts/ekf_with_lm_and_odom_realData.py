'''math tool'''
import csv
import math
import numpy as np
from numpy import cos, sin, arctan2
from scipy.spatial import distance as dist

'''plot tool'''
import matplotlib.pyplot as plt
from matplotlib import animation

'''image tool'''
import cv2
import statistics as sta

import sys

import utm
from pyproj import Proj

'''self module'''
import load_data_utils
import plot_utils
import EKF_localization
import ICP_correspondences

if __name__ == "__main__":

    file_path = '/home/ncslaber/110-1/210922_EKF-fusion-test/zigzag_bag/'
    filtered_map_x, filtered_map_y, filtered_map_t = load_data_utils.get_filtered_map_pose(file_path)
    filtered_utm_x, filtered_utm_y, filtered_utm_t = load_data_utils.get_GPS(file_path) 
    filtered_x, filtered_y, filtered_t = load_data_utils.get_filtered_pose(file_path)
        
    t = filtered_map_x.shape[1]
    t = np.arange(0, t, 1)
    t = np.reshape(t, (1,-1))

    ''' belief (estimate from EKF) '''
    mu_x = np.zeros(t.shape)
    mu_y = np.zeros(t.shape)
    mu_theta = np.zeros(t.shape)   # radians
    ''' starting belief - initial condition (robot pose of GPS)  
    lat, lng = (25.01728203283927, 121.54163935542573)
    _, _, zone, R = utm.from_latlon(lat, lng)
    proj = Proj(proj='utm', zone=zone, ellps='WGS84', preserve_units=False)  
    utm_x_init, utm_y_init = proj(lng, lat)'''
    utm_x_init, utm_y_init = 352848.75, 2767652.8 #2767652.5

    mu_x[0,99] = utm_x_init
    mu_y[0,99] = utm_y_init
    mu_theta[0,99] = filtered_map_t[0,99]
        # mu_x[0,0] = utm_x_init
        # mu_y[0,0] = utm_y_init
        # mu_theta[0,0] = filtered_map_t[0,0]
    '''initial uncertainty in the belief'''
    sigma = np.array( [[1, 0, 0],  # x
                       [0, 1, 0],  # y
                       [0, 0, .1] ]) # theta
    
    '''command velocity'''
    # v_c = 1 + 0.5*cos(2*np.pi*(0.2)*t)
    # omg_c = -0.2 + 2*cos(2*np.pi*(1.2)*t)
    '''noise in the command velocities (translational and rotational)'''
    # alpha = np.array([.1, .01, .01, .1])
    # alpha_1, alpha_2, alpha_3, alpha_4 = alpha

    '''landmarks'''
    lm_x, lm_y, lm_radi = load_data_utils.get_landmark_from_file()
    assert (len(lm_x)==len(lm_y))
    '''std deviation of range and bearing sensor noise for each landmark'''
    std_dev_dist = .5
    std_dev_phi = .5
    std_dev_radi = .5
    '''uncertainty due to measurement noise'''
    Q_t = np.array([ [std_dev_dist, 0, 0],
                     [0, std_dev_phi, 0],
                     [0, 0, std_dev_radi] ] )
    Q_t_tmp = np.array([ [std_dev_dist, 0, 0],
                     [0, std_dev_phi/20, 0],
                     [0, 0, std_dev_radi/20] ] )
    std_dev_state = .05
    R_t = np.array([ [std_dev_state, 0, 0],
                     [0, std_dev_state, 0],
                     [0, 0, std_dev_state] ] )

    ''' before update (here = last step) '''
    mu_hat_x = np.zeros(t.shape)
    mu_hat_y = np.zeros(t.shape)
    mu_hat_theta = np.zeros(t.shape)   # radians
    mu_hat_x[0,0] = mu_x[0,0]
    mu_hat_y[0,0] = mu_y[0,0]
    mu_hat_theta[0,0] = mu_theta[0,0]

    '''# of observed landmarks'''
    obs_lm_number = np.zeros(t.shape)
    cols=[]

    
    '''run EKF'''
    mu = np.array([ [mu_x[0,99]],[mu_y[0,99]],[mu_theta[0,99]] ])
    
    for i in range(100,3627):
        print(">>>>new iteration: "+str(i))
        flag = False
        icp_flag = 0
        cols=[]
        
        prev_theta = mu_theta[0,i-1]
        

        '''prediction step'''
        prev_odom_hat = np.array([ [filtered_x[0, i-1]], [filtered_y[0, i-1]], [filtered_t[0, i-1]] ])
        odom_hat = np.array([ [filtered_x[0, i]], [filtered_y[0, i]], [filtered_t[0, i]] ])
        # odom_hat += make_noise(R_t)
        # print((prev_odom_hat, odom_hat))

        mu_bar = EKF_localization.get_mu_bar_odom_modle(mu, (prev_odom_hat, odom_hat), north_heading=True)
        
        G_t = EKF_localization.get_G_t_odom((prev_odom_hat, odom_hat), prev_theta)
        sigma_bar = (G_t @ sigma @ (G_t.T)) #+ (V_t @ M_t @ (V_t.T))
        print('sigma_bar: ', sigma_bar)

        '''correction (updating belief based on landmark readings)'''
        bel_x = mu_bar[0,0]
        bel_y = mu_bar[1,0]
        bel_theta = mu_bar[2,0]

        '''measured landmarks'''
        np_z_true, obs_lm_utm_x, obs_lm_utm_y, obs_lm_radi = load_data_utils.get_observed_lm(file_path, i)
        print('np_z_true: ', np_z_true)

        np_z_hat=np.array([[],[],[]])

        ''' find correspondence'''
        if len(obs_lm_utm_x) > 0: 
            flag = True
            if len(obs_lm_utm_x) <= 2:
                if len(obs_lm_utm_x) == 0: ##########
                    ''' find correspondence by maximum likelihood '''
                    npLikelihood = np.array([])
                    list_z_hat, list_S_t, list_H_t = [],[],[]
                    print('z_true: ', z_true)
                    
                    print('>>>> find correspondence by maximum likelihood')
                    P = np.vstack((lm_x, lm_y))
                    U = np.vstack((obs_lm_utm_x, obs_lm_utm_y))
                    D = dist.cdist(U.T, P.T)
                    print('D: ',D)
                    cols = ICP_correspondences.get_correspondences_in_range(D[0], dist=2)
                    print('radius under 2m: ',cols)
                    if cols.shape[0] == 0:
                        print("under!")
                    for k in range(len(cols)):
                        
                        '''extract correspondences of ref map'''
                        m_j_x = lm_x[cols[k]]
                        m_j_y = lm_y[cols[k]]
                        m_j_radi = lm_radi[cols[k]]

                        '''get predict lm measurement'''
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
                    if npLikelihood.shape[0] ==0:
                        mu, sigma = mu_bar, sigma_bar
                        mu_x[0 , i] = mu[0 , 0]
                        mu_y[0 , i] = mu[1 , 0]
                        mu_theta[0 , i] = mu[2 , 0]
                        # obs_lm_number[0 , i] = len(np_z_hat)/3

                        mu_hat_x[0,i], mu_hat_y[0,i], mu_hat_theta[0,i] = bel_x, bel_y, bel_theta
                        continue

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
                    icp_flag = 1
                    K_t = sigma_bar @ (H_t.T) @ np.linalg.inv(S_t)
                    mu_bar = mu_bar+K_t@(z_true-z_hat)
                    sigma_bar = (np.identity(sigma_bar.shape[0])-(K_t @ H_t)) @ sigma_bar

                    '''save estimated measured data'''
                    np_z_hat=np.append(np_z_hat,z_hat)

                elif len(obs_lm_utm_x) == 2:
                    print('>>>> find correspondence by maximum likelihood 2')
                    P = np.vstack((lm_x, lm_y))
                    U = np.vstack((obs_lm_utm_x, obs_lm_utm_y))
                    D = dist.cdist(U.T, P.T)
                    print('D: ',D)

                    '''iterative update predict measurement'''
                    for tmp in [0,1]:
                        ''' find correspondence by maximum likelihood '''
                        npLikelihood = np.array([])
                        list_z_hat = []
                        list_S_t = []
                        list_H_t = []
                        cols = ICP_correspondences.get_correspondences_in_range(D[tmp], dist=2)
                        print('cols: ', cols)
                        if cols.shape[0] == 0:
                            print("under!")
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
                        if npLikelihood.shape[0] ==0:
                            mu, sigma = mu_bar, sigma_bar
                            mu_x[0 , i] = mu[0 , 0]
                            mu_y[0 , i] = mu[1 , 0]
                            mu_theta[0 , i] = mu[2 , 0]
                            # obs_lm_number[0 , i] = len(np_z_hat)/3

                            mu_hat_x[0,i], mu_hat_y[0,i], mu_hat_theta[0,i] = bel_x, bel_y, bel_theta
                            continue
                        maxLikelihood = npLikelihood.argmax()
                        print('npLikelihood: ',npLikelihood)
                        print("matched lm: ", cols[maxLikelihood])
                        H_t = list_H_t.pop(maxLikelihood)
                        S_t = list_S_t.pop(maxLikelihood)
                        z_hat = list_z_hat.pop(maxLikelihood)
                                               
                        '''kalman gain and update belief'''
                        icp_flag = 2
                        K_t = sigma_bar @ (H_t.T) @ np.linalg.inv(S_t)
                        z_true = np.reshape(np_z_true[:,tmp], (-1,3))
                        z_true = z_true.T
                        mu_bar = mu_bar+K_t@(z_true-z_hat)
                        sigma_bar = (np.identity(sigma_bar.shape[0])-(K_t @ H_t)) @ sigma_bar

                        '''save estimated measured data'''
                        np_z_hat=np.append(np_z_hat,z_hat)

            # else:
            #     ''' find correspondence by ICP '''
            #     print('>>>> find correspondence by ICP')
            #     P = np.vstack((lm_x, lm_y))
            #     U = np.vstack((obs_lm_utm_x, obs_lm_utm_y)) #############(obs_lm_x, obs_lm_y)
                
            #     cols = ICP_correspondences.get_Rt_by_ICP(P,U)
            #     if cols.shape[0] == 0:
            #         print("under!")
            #         mu, sigma = mu_bar, sigma_bar
            #         mu_x[0 , i] = mu[0 , 0]
            #         mu_y[0 , i] = mu[1 , 0]
            #         mu_theta[0 , i] = mu[2 , 0]
            #         # obs_lm_number[0 , i] = len(np_z_hat)/3

            #         mu_hat_x[0,i], mu_hat_y[0,i], mu_hat_theta[0,i] = bel_x, bel_y, bel_theta
            #         continue
            #     '''iterative update predict measurement'''
            #     for k in range(len(obs_lm_utm_x)):
            #         m_j_x = lm_x[cols[k]]
            #         m_j_y = lm_y[cols[k]]
            #         m_j_radi = lm_radi[cols[k]]
                                                    
            #         '''get predict measurement'''
            #         diff_x = m_j_x - bel_x
            #         diff_y = m_j_y - bel_y
                    
            #         z_hat, H_t, S_t \
            #             = EKF_localization.get_predict_lm_measure(diff_x, diff_y, bel_theta, m_j_radi, sigma_bar, Q_t)
                                        
            #         '''kalman gain and update belief'''
            #         icp_flag = 3
            #         K_t = sigma_bar @ (H_t.T) @ np.linalg.inv(S_t)
            #         z_true = np.reshape(np_z_true[:,k], (-1,3))
            #         z_true = z_true.T
            #         print('z_true-z_hat: ',z_true-z_hat)
            #         mu_bar = mu_bar+K_t@(z_true-z_hat)
            #         sigma_bar = (np.identity(sigma_bar.shape[0])-(K_t @ H_t)) @ sigma_bar

            #         '''save estimated measured data'''
            #         np_z_hat=np.append(np_z_hat,z_hat)
            np_z_hat = np.reshape(np_z_hat, (-1,3))
            np_z_hat = np_z_hat.T

        
        '''update by GPS'''
        z_true = np.array([ [-filtered_map_y[0, i]],
                            [filtered_map_x[0, i]],
                            [filtered_map_t[0, i]]
                          ])
        print("KF-z_true: ", z_true)
        C = np.array([  [1,0,0],
                        [0,1,0],
                        [0,0,1]
                     ])
        S_t = (C @ sigma_bar @ (C.T)) + R_t
        K_t = sigma_bar @ (C.T) @ np.linalg.inv(S_t)
        z_hat = mu_bar - np.array([ [mu_x[0,99]], [mu_y[0,99]], [0] ])
        print("KF-z_hat: ", z_hat)
        mu_bar = mu_bar+K_t@(z_true-z_hat)
        print("KF-(z_true-z_hat): ", z_true-z_hat)
        sigma_bar = (np.identity(sigma_bar.shape[0])-(K_t @ C)) @ sigma_bar

        '''update belief'''
        mu, sigma = mu_bar, sigma_bar
        
        mu_x[0 , i] = mu[0 , 0]
        mu_y[0 , i] = mu[1 , 0]
        mu_theta[0 , i] = mu[2 , 0]
        # obs_lm_number[0 , i] = len(np_z_hat)/3
        print('UPDATED mu: ', mu)

        mu_hat_x[0,i], mu_hat_y[0,i], mu_hat_theta[0,i] = bel_x, bel_y, bel_theta
         
        '''check remapped landmark pose
        obs_lm_x, obs_lm_y, obs_lm_radi = load_data_utils.get_observed_lm_relative_pose(file_path, i)'''

        print("<<<<finish update: "+str(i))

        # if  flag==True: #len(obs_lm_x) != 0:
        #     # print("difference btw mu_bar and robot_xy_new: ", mu_bar, robot_xy_new)
        #     # tmp = (mu_bar[:2,0] - robot_xy_new[:,0])
        #     plot_traj((x_pos_true, y_pos_true, theta_pos_true), (mu_x, mu_y, mu_theta), (obs_lm_x, obs_lm_y, obs_lm_radi),(lm_x,lm_y,lm_radi),i, \
        #                 np_z_hat, np_z_true, (bel_x, bel_y, bel_theta), (real_x, real_y, real_tehta), cols, icp_flag, flag)   
        
        if i%1200 == 0:
            plot_utils.plot_traj( (filtered_utm_x,filtered_utm_y,filtered_utm_t), (mu_x, mu_y, mu_theta), (obs_lm_utm_x, obs_lm_utm_y, obs_lm_radi), (lm_x,lm_y,lm_radi),i, \
                np_z_hat, np_z_true, (bel_x, bel_y, bel_theta), mu, cols, icp_flag)
            # plot_utils.plot_traj( (mu_hat_x, mu_hat_y, mu_hat_theta),(utm_x_loc_origin, utm_y_loc_origin, utm_t_loc_origin), (mu_x, mu_y, mu_theta), (obs_lm_utm_x, obs_lm_utm_y, obs_lm_radi),(lm_x,lm_y,lm_radi),i, \
            #         np_z_hat, np_z_true, cols, icp_flag, flag) 
        # plot_traj(gps_states, belief_states, update_states, markers, markers_in_map, index, np_z_hat, np_z_true, cols, icp_flag, flag):
    # np.save('/home/ncslaber/class_material/EKF_localization_with_unknown_correspondences/data_ground_truth/mu_x', mu_x)
    # np.save('/home/ncslaber/class_material/EKF_localization_with_unknown_correspondences/data_ground_truth/mu_y', mu_y)
    # np.save('/home/ncslaber/class_material/EKF_localization_with_unknown_correspondences/data_ground_truth/mu_theta', mu_theta)
    # np.save('/home/ncslaber/class_material/EKF_localization_with_unknown_correspondences/data_ground_truth/obs_lm_number', obs_lm_number)
    # np.save('/home/ncslaber/class_material/EKF_localization_with_unknown_correspondences/data_ground_truth/obs_lm_vector', np_z_true)
    