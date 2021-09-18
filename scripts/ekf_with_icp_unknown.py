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

def plot_traj(true_states, belief_states, markers, markers_in_map, index):
    x_tr, y_tr, th_tr = true_states
    x_guess, y_guess = belief_states

    radius = 0.5
    
    world_bounds = [-10,10]
    fig, ax = plt.subplots(figsize=(10,10),dpi=120)
    ax = plt.axes(xlim=world_bounds, ylim=world_bounds)
    ax.set_aspect('equal')

    '''plot landmarkers'''
    plt.scatter(markers_in_map[0], markers_in_map[1], marker='X',s=100, color='g', label='ref landmarks')
    number_of_point=12
    piece_rad = np.pi/(number_of_point/2)
    
    for j in range( len(markers_in_map[0]) ):
        neg_bd = []
        for i in range(number_of_point):
            neg_bd.append((markers_in_map[0][j]+markers_in_map[2][j]*np.cos(piece_rad*i), markers_in_map[1][j]+markers_in_map[2][j]*np.sin(piece_rad*i)))
        neg_bd=np.asarray(neg_bd)
        plt.scatter(neg_bd[:,0], neg_bd[:,1], c='k', s=10)

    '''plot traj'''
    plt.scatter(x_tr[0], y_tr[0], color='b', label="Actual", s=10)
    plt.scatter(x_guess[0], y_guess[0], color='r', label="Predicted", s=10)

    '''plot final state'''
    plt.scatter(x_tr[0][index],y_tr[0][index], s=500, color='y', ec='k')
    plt.plot( [x_tr[0][index], x_tr[0][index] + radius*cos(th_tr[0][index]) ], 
               [y_tr[0][index], y_tr[0][index] + radius*sin(th_tr[0][index]) ], color='k' )

    '''plot obs lm'''
    for i in range( len(markers[0]) ):
        plt.plot([ markers[0][i],x_guess[0][index] ],
                [ markers[1][i],y_guess[0][index] ], color='k')
    plt.title('update times: '+str(index)+'/200', fontsize=20)
    plt.legend()
    plt.show()

def get_mu_bar(prev_mu, velocity, omega, angle, dt):
    ratio = velocity/omega
    m = np.array([[(-ratio*sin(angle))+(ratio*sin(angle+omega*dt))],
                  [(ratio*cos(angle))-(ratio*cos(angle+omega*dt))],
                  [omega*dt]])
    return prev_mu + m

def get_observed_lm(mu_bar, global_lm):
    obs_lm_x = []
    obs_lm_y = []
    obs_lm_radi = []
    FOV = np.pi/2
    for i in range( len(global_lm[0]) ):
        diff_x = global_lm[0][i] - mu_bar[0]
        diff_y = global_lm[1][i] - mu_bar[1]
        diff_theta = arctan2(diff_y, diff_x) - mu_bar[2]

        if diff_theta > -FOV and diff_theta < FOV:
            if diff_x*diff_x+diff_y*diff_y < 49:
                obs_lm_x.append( global_lm[0][i])
                obs_lm_y.append( global_lm[1][i])
                obs_lm_radi.append( global_lm[2][i])
                
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

def find_second_min(row_index, D, cols):
    second_min = 100
    sm_index = 0
    row_list = D[row_index,:]
    for i in range(len(row_list)):
        if i != cols[row_index]:
            if row_list[i] < second_min:
                second_min = row_list[i]
                sm_index = i
    
    return second_min, sm_index

def get_correspondences(P,U):
    
    D = dist.cdist(U.T, P.T)
    print('>>>> start to ICP >>>>')
    print('dist: ', D)
    # rows = D.min(axis=1)
    cols = D.argmin(axis=1)
    print('match point: ', list(enumerate(cols)))

    for a in range(len(cols)):
        b=1
        while a+b < len(cols):
            if cols[a] == cols[a+b]:
                print("matched same ref landmarks!")
                second_min_A,A = find_second_min(a, D, cols)
                second_min_B,B = find_second_min(a+b, D, cols)
                if second_min_A>second_min_B:
                    cols[a+b]=B
                elif second_min_A<second_min_B:
                    cols[a]=A
                print('changed matched point-set: ', list(enumerate(cols)))
            b+=1
        
    Q = np.zeros(U.shape)
    for (row, col) in enumerate(cols):
        Q[:,row] = P[:,col]

    return Q, cols

def get_Rt_by_ICP(P,U, robot_xy):
    print('U: ',U)
    resid_scalar = 50
    count = 0
    while resid_scalar > 1:
        count += 1
        Q, cols = get_correspondences(P,U)
        U_bar = np.array([np.average(U, axis=1)])
        U_bar = U_bar.T
        Q_bar = np.array([np.average(Q, axis=1)])
        Q_bar = Q_bar.T

        X = U-U_bar
        Y = Q-Q_bar
        S = X @ Y.T
        u, s, vh = np.linalg.svd(S)

        det = np.linalg.det(vh@u.T)
        print('det(vh@u.T): ', det)
        if det>0:
            tmp = np.array([ [1,0],[0,1] ])
        else: 
            tmp = np.array([ [1,0],[0,-1] ])
        R = vh @ tmp @ u.T
        t = Q_bar-U_bar
        print('R: ', R)
        print('t: ', t, Q_bar, U_bar)
        U_new = R@X + U_bar+t
        robot_xy_decentral = robot_xy-U_bar
        robot_xy_new = R @ robot_xy_decentral + (U_bar+t)

        # calculate residuals
        residuals = Q-U_new
        residuals = np.absolute(residuals)
        resid_scalar = residuals.sum()
        print("residual = ",resid_scalar)
        U = U_new
        robot_xy = np.array([[robot_xy_new[0][0]],[robot_xy_new[1][0]]])
        print("iteration time: ", count)
        if count>4:
            print("iterate over 5 times!!")
            break

    return robot_xy_new, R,t, Q, cols 

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
    lm_x = [6,-7,6,2,3,5]
    lm_y = [4,8,-4,7,6,5]
    lm_radi = [0.2, 0.5, 0.3, 0.2, 0.5, 0.3]
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
    x_pos_true = np.load('/home/ncslaber/class_material/EKF_localization_with_unknown_correspondences/data_ground_truth/x_pos_true.npy')
    y_pos_true = np.load('/home/ncslaber/class_material/EKF_localization_with_unknown_correspondences/data_ground_truth/y_pos_true.npy')
    theta_pos_true = np.load('/home/ncslaber/class_material/EKF_localization_with_unknown_correspondences/data_ground_truth/theta_pos_true.npy')
    # x_pos_true = np.zeros(t.shape)
    # y_pos_true = np.zeros(t.shape)
    # theta_pos_true = np.zeros(t.shape)
    '''set ground truth data by calculation
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
    np.save('/home/ncslaber/class_material/EKF_localization_with_unknown_correspondences/data_ground_truth/x_pos_true', x_pos_true)
    np.save('/home/ncslaber/class_material/EKF_localization_with_unknown_correspondences/data_ground_truth/y_pos_true', y_pos_true)
    np.save('/home/ncslaber/class_material/EKF_localization_with_unknown_correspondences/data_ground_truth/theta_pos_true', theta_pos_true)'''
    # np.save('/home/ncslaber/class_material/EKF_localization_with_unknown_correspondences/data_ground_truth/theta_pos_true', velocity)
    # np.save('/home/ncslaber/class_material/EKF_localization_with_unknown_correspondences/data_ground_truth/theta_pos_true', omega)

    '''run KF'''
    mu = np.array([ [mu_x[0,0]],[mu_y[0,0]],[mu_theta[0,0]] ])
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
        mu_bar = get_mu_bar(mu, curr_v, curr_w, prev_theta, dt)
        # mu = np.array([ [x_pos_true[0,i]],[y_pos_true[0,i]],[theta_pos_true[0,i]] ])
        sigma_bar = (G_t @ sigma @ (G_t.T)) + (V_t @ M_t @ (V_t.T))

        '''correction (updating belief based on landmark readings)'''
        bel_x = mu_bar[0,0]
        bel_y = mu_bar[1,0]
        bel_theta = mu_bar[2,0]
        # cannot observe all lm !
        obs_lm_x, obs_lm_y, obs_lm_radi= get_observed_lm(mu_bar, (lm_x, lm_y, lm_radi))
        
        ''' find correspondence by ICP '''
        if len(obs_lm_x) != 0:
            real_x = x_pos_true[0,i]
            real_y = y_pos_true[0,i]
            real_tehta = theta_pos_true[0,i]
            P = np.vstack((lm_x, lm_y))
            U = np.vstack((obs_lm_x, obs_lm_y))
            robot_xy = np.array([ [real_x],
                                [real_y] ])
            robot_xy_new,R,t,Q, cols = get_Rt_by_ICP(P,U, robot_xy)
        
        for k in range(len(obs_lm_x)):
            npLikelihood = np.array([])
            list_z_hat = []
            list_S_t = []
            list_H_t = []
            
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
            z_true += make_noise(Q_t)
            
            # get_nearby_tree

            m_j_x = lm_x[cols[k]]
            m_j_y = lm_y[cols[k]]
            m_j_radi = lm_radi[cols[k]]
                                            
            '''likelihood'''
            diff_x = m_j_x - bel_x
            diff_y = m_j_y - bel_y
            q = (diff_x ** 2) + (diff_y ** 2)
            z_hat = np.array([ [np.sqrt(q)],
                                [arctan2(diff_y, diff_x) - bel_theta],
                                [m_j_radi] ])
            H_t = np.array([ [-diff_x / np.sqrt(q), -diff_y / np.sqrt(q), 0],
                                [diff_y / q, -diff_x / q, -1],
                                [0,0,0] ])
            S_t = (H_t @ sigma_bar @ (H_t.T)) + Q_t
            # likelihood = np.sqrt(np.linalg.det(2*np.pi*S_t)) * math.exp(-0.5*((z_true-z_hat).T)@(np.linalg.inv(S_t))@(z_true-z_hat))
            # npLikelihood = np.append(npLikelihood,likelihood)
            
            # list_z_hat.append(z_hat)
            # list_S_t.append(S_t)
            # list_H_t.append(H_t)

            '''maximum likelihood'''
            # maxLikelihood = npLikelihood.argmax()
            # H_t = list_H_t.pop(maxLikelihood)
            # S_t = list_S_t.pop(maxLikelihood)
            # z_hat = list_z_hat.pop(maxLikelihood)
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

        if len(obs_lm_x) != 0:
            print("difference btw mu_bar and robot_xy_new: ", mu_bar, robot_xy_new)
            tmp = (mu_bar[:2,0] - robot_xy_new[:,0])
            # if tmp[0]*tmp[0] + tmp[1]*tmp[1] > 0.5: 
            plot_traj((x_pos_true, y_pos_true, theta_pos_true), (mu_x, mu_y), (obs_lm_x, obs_lm_y, obs_lm_radi),(lm_x,lm_y,lm_radi),i) 
        
        if i%50 == 0:
            plot_traj((x_pos_true, y_pos_true, theta_pos_true), (mu_x, mu_y), (obs_lm_x, obs_lm_y, obs_lm_radi),(lm_x,lm_y,lm_radi),i) 