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

import utm
from pyproj import Proj

# np.load("/home/ncslaber/110-1/210922_EKF-fusion-test/zigzag_bag/depth/" + str(self.timestampSecs%1000)+'-'+str(int(self.timestampNSecs/1e6)), self.imgDepth)
# from EKF_localization import localize


def plot_traj(belief_states, gps_states, update_states, markers, markers_in_map, index, np_z_hat, np_z_true, cols, icp_flag, flag):
    x_tr, y_tr, th_tr = gps_states
    x_init_guess, y_init_guess, theta_init_guess = belief_states   
    x_update, y_update, theta_update = update_states

    # x_tr_lm, y_tr_lm, th_tr_lm = lm_states

    radius = 0.5
    
    # world_bounds = [-15,10]
    fig, ax = plt.subplots(figsize=(10,10))
    # ax = plt.axes(xlim=world_bounds, ylim=world_bounds)
    ax.set_aspect('equal')

    '''plot landmarkers'''
    plt.scatter(markers_in_map[0], markers_in_map[1], marker='X',s=100, color='g', label='ref landmarks')
    plt.scatter(markers[0], markers[1], marker='X',s=100, color='r', label='gps_obs landmarks')
    number_of_point=12
    piece_rad = np.pi/(number_of_point/2)
    for j in range( len(markers_in_map[0]) ):
        neg_bd = []
        for i in range(number_of_point):
            neg_bd.append((markers_in_map[0][j]+markers_in_map[2][j]*np.cos(piece_rad*i), markers_in_map[1][j]+markers_in_map[2][j]*np.sin(piece_rad*i)))
        neg_bd=np.asarray(neg_bd)
        plt.scatter(neg_bd[:,0], neg_bd[:,1], c='k', s=10)
    for j in range( len(markers[0]) ):
        neg_bd = []
        for i in range(number_of_point):
            neg_bd.append((markers[0][j]+markers[2][j]*np.cos(piece_rad*i), markers[1][j]+markers[2][j]*np.sin(piece_rad*i)))
        neg_bd=np.asarray(neg_bd)
        plt.scatter(neg_bd[:,0], neg_bd[:,1], c='k', s=10)

    '''plot traj'''
    plt.scatter(x_tr[0][:index], y_tr[0][:index], color='b', label="gps_states", s=10)
    # plt.scatter(x_init_guess, y_init_guess, color='b', label="gps_states", s=10)
    plt.scatter(x_update[0][:index], y_update[0][:index], color='g', label="update_states", s=10)
          
    '''plot final state'''
    plt.scatter(x_tr[0][index],y_tr[0][index], s=300, color='lightblue', ec='k', label='GPS pose')
    plt.plot( [x_tr[0][index], x_tr[0][index] + radius*cos(th_tr[0][index]) ], 
                [y_tr[0][index], y_tr[0][index] + radius*sin(th_tr[0][index]) ], color='k' )
    plt.scatter(x_update[0][index],y_update[0][index], s=500, color='y', ec='k', label='Predicted pose')
    plt.plot( [x_update[0][index], x_update[0][index] + radius*cos(theta_update[0][index]) ], 
                [y_update[0][index], y_update[0][index] + radius*sin(theta_update[0][index]) ], color='k' )
    # plt.scatter(x_tr_lm[0][index],y_tr_lm[0][index], s=400, color='lightblue', ec='k', label='landmark pose')
    # plt.plot( [x_tr_lm[0][index], x_tr_lm[0][index] + radius*cos(th_tr_lm[0][index]) ], 
    #           [y_tr_lm[0][index], y_tr_lm[0][index] + radius*sin(th_tr_lm[0][index]) ], color='k' )

    '''plot observation z'''
    plot_measured_landmarks(np_z_hat, np_z_true, (x_init_guess[0][index], y_init_guess[0][index], theta_init_guess[0][index]), \
                                                    (x_tr[0][index], y_tr[0][index], th_tr[0][index]), \
                                                        (x_update[0][index], y_update[0][index], theta_update[0][index]))
    if flag == True:
        if icp_flag == True:
            plt.text(3.0, 8, 'icp matched points: '+str(cols),fontsize=14)
        else:
            plt.text(3.0, 8, 'maximum matched point',fontsize=14)

    '''plot obs lm
    for i in range( len(markers[0]) ):
        plt.plot([ markers[0][i],x_tr[0][index] ],
                [ markers[1][i],y_tr[0][index] ], color='k')'''
    plt.scatter(x_update[0][:index], y_update[0][:index], color='r', label="Predicted", s=10)
    plt.title('update times: '+str(index)+'/200', fontsize=25)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.legend(fontsize=15)
    plt.show()
    # fig.savefig('/home/ncslaber/110-1/210922_EKF-fusion-test/zigzag_bag/TMP'+str(index)+'.png')

def plot_measured_landmarks(np_z_hat, np_z_true, bel_pose, gps_pose, update_pose):
    bel_x, bel_y, bel_theta = bel_pose
    gps_x, gps_y, gps_theta = gps_pose
    update_x, update_y, update_theta = update_pose
    radius = 0.5

    # world_bounds = [-15,10]
    # fig, ax = plt.subplots(figsize=(10,10),dpi=120)
    # ax = plt.axes(xlim=world_bounds, ylim=world_bounds)
    # ax.set_aspect('equal')

    '''plot state'''
    # bel_theta -= np.pi/2
    # gps_theta -= np.pi/2
    plt.scatter(bel_x, bel_y, s=300, color='lightyellow', ec='k', label='z_hat pose')
    plt.plot( [bel_x, bel_x + radius*cos(bel_theta) ], 
               [bel_y, bel_y + radius*sin(bel_theta) ], color='k' )
    
    '''plot observation'''
    
    for i in range( np_z_true.shape[1] ):
        r_x = np_z_hat[0][i] * cos(np_z_hat[1][i])
        r_y = np_z_hat[0][i] * sin(np_z_hat[1][i])
        wr_x =  cos(bel_theta)*r_x - sin(bel_theta)*r_y
        wr_x += bel_x
        wr_y = sin(bel_theta)*r_x + cos(bel_theta)*r_y
        wr_y += bel_y
        plt.plot([ bel_x,wr_x ],
                 [ bel_y,wr_y ], color='k', linestyle='--', label='predi_observing')

        r_x = np_z_true[0][i] * cos(np_z_true[1][i])
        r_y = np_z_true[0][i] * sin(np_z_true[1][i])
        wr_x =  cos(gps_theta)*r_x - sin(gps_theta)*r_y
        wr_x += gps_x
        wr_y = sin(gps_theta)*r_x + cos(gps_theta)*r_y
        wr_y += gps_y
        plt.plot([ gps_x,wr_x ],
                 [ gps_y,wr_y ], color='b', linestyle='-', label='gps_observing')

def plot_transformed(P, U, robot_pose, theta, count):
    # robot_x, robot_y, robot_theta = robot_pose
    radius = 0.5

    world_bounds = [-15,10]
    fig, ax = plt.subplots(figsize=(10,10),dpi=120)
    ax = plt.axes(xlim=world_bounds, ylim=world_bounds)
    ax.set_aspect('equal')

    '''plot state'''
    plt.scatter(robot_pose[0,0], robot_pose[1,0], s=300, color='lightblue', ec='k', label='robot_true pose')
    plt.plot( [robot_pose[0,0], robot_pose[0,0] + radius*cos(theta) ], 
              [robot_pose[1,0], robot_pose[1,0] + radius*sin(theta) ], color='k' )
    
    '''plot transformed observation'''
    plt.scatter(P[0,:], P[1,:], c='g',s=300, marker="X",label='ref lms')
    plt.scatter(U[0,:], U[1,:], c='r',s=300, label='transformed lms')

    plt.title('process of transforming, iteration time: '+str(count), fontsize=25)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.legend(fontsize=15)
    plt.show()
    
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
                if second_min_A>=second_min_B:
                    cols[a+b]=B
                elif second_min_A<second_min_B:
                    cols[a]=A
                print('changed matched point-set: ', list(enumerate(cols)))
            b+=1
        
    Q = np.zeros(U.shape)
    for (row, col) in enumerate(cols):
        Q[:,row] = P[:,col]

    return Q, cols

def get_Rt_by_ICP(P,U, robot_xy, theta):
    resid_scalar = 50
    count = 0
    '''plot transformed result'''
    # plot_transformed(P,U, robot_xy,theta, count)
    print('>>>> start to ICP >>>>')
    while resid_scalar > 1:
        print(">>iteration time: ", count)
        print('P: ',P)
        print('U: ',U)
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
        print('t: ', t)
        print('Q_bar: ', Q_bar)
        print('U_bar: ', U_bar)
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
        
        '''plot transformed result'''
        # plot_transformed(P,U, robot_xy, theta, count)

        if count>4:
            print("iterate over 5 times!!")
            break

    return cols 

def get_landmark():

    lm_x = []
    lm_y = []
    lm_radi = []
    ''' load landmark map '''
    directory = '/home/ncslaber/mapping_node/mapping_ws/src/mapping_explorer/NTU_allMaps/'
    files_to_check = ['210906_loopClosure']
    file_path_map = directory+files_to_check[0]
    shp_path = file_path_map + '/shapefiles/'
    for i in range(0,2):
        center = np.load(shp_path+'center_'+str(i+1)+'_bd_utm.npy')
        if center is None:
            print("neg_bd is empty!!")
        else: 
            lm_x.append(center[0])
            lm_y.append(center[1])
            lm_radi.append(center[2])
    # center_utm_ref = np.load(file_path_map+'center_utm_ref.npy')
    # lm_x = [-7,2,3,5,6,6,3] 
    # lm_y = [8,7,6,5,4,-4,-6] 
    # lm_radi = [0.2, 0.5, 0.3, 0.2, 0.5, 0.3, 0.4]
    
    
    return lm_x, lm_y, lm_radi

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
    # dt = .1
    # t = np.arange(0, 20+dt, dt)
    # t = np.reshape(t, (1,-1))
    file_path = '/home/ncslaber/110-1/210922_EKF-fusion-test/zigzag_bag/'
    with open(file_path + 'cb_pose.csv', 'r') as csvfile:
        robot_pose_gps = list( csv.reader(csvfile, delimiter=',') )
        robot_pose_gps = np.array(robot_pose_gps).astype(float)
        
    t = robot_pose_gps.shape[0]
    t = np.arange(0, t, 1)
    t = np.reshape(t, (1,-1))

    ''' belief (estimate from EKF) '''
    mu_x = np.zeros(t.shape)
    mu_y = np.zeros(t.shape)
    mu_theta = np.zeros(t.shape)   # radians
    ''' starting belief - initial condition (robot pose) '''
    lat = robot_pose_gps[:,0]
    lng = robot_pose_gps[:,1]
    _, _, zone, R = utm.from_latlon(lat, lng)
    proj = Proj(proj='utm', zone=zone, ellps='WGS84', preserve_units=False)
    utm_x_loc_origin, utm_y_loc_origin = proj(lng, lat)
    utm_x_loc_origin.shape = (1,-1)
    utm_y_loc_origin.shape = (1,-1)
    utm_t_loc_origin = robot_pose_gps[:,2]
    utm_t_loc_origin.shape = (1,-1)

    mu_x[0,0] = utm_x_loc_origin[0,0]
    mu_y[0,0] = utm_y_loc_origin[0,0]
    print('utm_x_loc_origin[0,0]: ', utm_x_loc_origin[0,0])
    print('utm_x_loc_origin shape: ', utm_x_loc_origin.shape)
    mu_theta[0,0] = utm_t_loc_origin[0,0]
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
    lm_x, lm_y, lm_radi = get_landmark()
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
    std_dev_state = .5
    R_t = np.array([ [std_dev_state, 0, 0],
                     [0, std_dev_state, 0],
                     [0, 0, std_dev_state] ] )

    '''ground truth
    x_pos_true = np.load('/home/ncslaber/class_material/EKF_localization_with_unknown_correspondences/data_ground_truth/x_pos_true.npy')
    y_pos_true = np.load('/home/ncslaber/class_material/EKF_localization_with_unknown_correspondences/data_ground_truth/y_pos_true.npy')
    theta_pos_true = np.load('/home/ncslaber/class_material/EKF_localization_with_unknown_correspondences/data_ground_truth/theta_pos_true.npy')'''
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
    
    ''' before update (estimate from EKF) '''
    mu_x_hat = np.zeros(t.shape)
    mu_y_hat = np.zeros(t.shape)
    mu_theta_hat = np.zeros(t.shape)   # radians
    mu_x_hat[0,0] = utm_x_loc_origin[0,0]
    mu_y_hat[0,0] = utm_y_loc_origin[0,0]
    mu_theta_hat[0,0] = utm_t_loc_origin[0,0]

    '''# of observed landmarks'''
    obs_lm_number = np.zeros(t.shape)
    cols=[]

    
    '''run EKF'''
    mu = np.array([ [mu_x[0,0]],[mu_y[0,0]],[mu_theta[0,0]] ])
    mu_bar = mu
    sigma_bar = sigma
    for i in range(1, 500):
        print(">>>>new ietration: "+str(i))
        flag = False
        icp_flag = False
        
        # This is only for temperary covariance
            # curr_v = v_c[0,i]
            # curr_w = omg_c[0,i]
            # prev_theta = mu_theta[0,i-1]

            # G_t = get_G_t(curr_v, curr_w, prev_theta, dt)
            # V_t = get_V_t(curr_v, curr_w, prev_theta, dt)
            # M_t = get_M_t(alpha, curr_v, curr_w)

        '''prediction step'''
        # mu_bar = get_mu_bar(mu, curr_v, curr_w, prev_theta, dt)
        
        # sigma_bar = (G_t @ sigma @ (G_t.T)) + (V_t @ M_t @ (V_t.T))

        '''correction (updating belief based on landmark readings)'''
        bel_x = mu_bar[0,0]
        bel_y = mu_bar[1,0]
        bel_theta = mu_bar[2,0]

        mu_x_hat[0,i] = bel_x
        mu_y_hat[0,i] = bel_y
        mu_theta_hat[0,i] = bel_theta

        '''measured landmarks
        # real_x = x_pos_true[0,i]
        # real_y = y_pos_true[0,i]
        # real_tehta = theta_pos_true[0,i]
        obs_lm_x, obs_lm_y, obs_lm_radi= get_observed_lm( [real_x,real_y,real_tehta], (lm_x, lm_y, lm_radi) )'''
        np_z_hat=np.array([[],[],[]])
        '''generate noise measurement
        np_z_true=np.array([[],[],[]])
        np_z_hat=np.array([[],[],[]])
        np_obs_x_m_noise=np.array([])
        np_obs_y_m_noise=np.array([])'''

        # for k in range(len(obs_lm_x)):
                       
        #     obs_k_x = obs_lm_x[k]
        #     obs_k_y = obs_lm_y[k]
        #     obs_k_radi = obs_lm_radi[k]
            
        '''get the sensor measurement
            diff_x = obs_k_x - real_x
            diff_y = obs_k_y - real_y
            q = (diff_x ** 2) + (diff_y ** 2)
            z_true = np.array([ [np.sqrt(q)],
                                [arctan2(diff_y, diff_x) - real_tehta],
                                [obs_k_radi] ])'''
            

        '''save real measured data
            np_z_true=np.append(np_z_true, z_true)
            obs_lm_x_noise, obs_lm_y_noise = get_noise_landmark_xy(z_true, (real_x, real_y, real_tehta))
            np_obs_x_m_noise=np.append(np_obs_x_m_noise,obs_lm_x_noise)
            np_obs_y_m_noise=np.append(np_obs_y_m_noise,obs_lm_y_noise)'''
        
        np_z_true, np_obs_x_m_noise, np_obs_y_m_noise, obs_lm_radi = get_observed_lm(i)
        np_z_true = np.reshape(np_z_true, (-1,3))
        np_z_true = np_z_true.T
        print('np_z_true: ', np_z_true)
        
        ''' find correspondence'''
        if len(np_obs_x_m_noise) > 0:
            flag = True
            if len(np_obs_x_m_noise) <= 2:
                if len(np_obs_x_m_noise) == 1:
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
                    bool_D = D[0]<2  # distance less than 2m
                    print('bool_D: ',bool_D)
                    cols = np.where(bool_D==True)
                    cols = cols[0]
                    print('radius under 1m: ',cols)
                    for k in range(len(cols)):
                        
                        m_j_x = lm_x[cols[k]]
                        m_j_y = lm_y[cols[k]]
                        m_j_radi = lm_radi[cols[k]]

                        '''get predict measurement'''
                        diff_x = m_j_x - bel_x
                        diff_y = m_j_y - bel_y
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
                        
                        likelihood = np.sqrt(np.linalg.det(2*np.pi*S_t)) * math.exp(-0.5*((z_true-z_hat).T)@(np.linalg.inv(S_t))@(z_true-z_hat))
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
                    print('real: ',(real_x, real_y, real_tehta))
                    H_t = list_H_t.pop(maxLikelihood)
                    S_t = list_S_t.pop(maxLikelihood)
                    z_hat = list_z_hat.pop(maxLikelihood)

                    '''save estimated measured data'''
                    np_z_hat=np.append(np_z_hat,z_hat)
                    
                    '''kalman gain and update belief'''
                    K_t = sigma_bar @ (H_t.T) @ np.linalg.inv(S_t)
                    mu_bar = mu_bar+K_t@(z_true-z_hat)
                    sigma_bar = (np.identity(sigma_bar.shape[0])-(K_t @ H_t)) @ sigma_bar
                elif len(np_obs_x_m_noise) == 2:
                    print('>>>> find correspondence by maximum likelihood 2')
                    P = np.vstack((lm_x, lm_y))
                    U = np.vstack((np_obs_x_m_noise, np_obs_y_m_noise))
                    D = dist.cdist(U.T, P.T)
                    print('D: ',D)

                    '''iterative update predict measurement'''
                    for tmp in [0,1]:
                        ''' find correspondence by maximum likelihood '''
                        npLikelihood = np.array([])
                        list_z_hat = []
                        list_S_t = []
                        list_H_t = []
                        bool_D = D[tmp]<2  # distance less than 2m
                        print('bool_D(matched ID of lms): ',bool_D)
                        cols = np.where(bool_D==True)
                        cols = cols[0]
                        print('cols: ', cols)
                        for k in range(len(cols)):
                            
                            m_j_x = lm_x[cols[k]]
                            m_j_y = lm_y[cols[k]]
                            m_j_radi = lm_radi[cols[k]]

                            '''get predict measurement'''
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
                            
                            z_true = np.reshape(np_z_true[:,tmp], (-1,3))
                            z_true = z_true.T
                            print('z_true:', z_true)
                            print('z_hat:', z_hat)
                            print('z_true-z_hat:', z_true-z_hat)
                            diff_z = z_true-z_hat
                            diff_z[2,0] *= 2
                            print('z_true-z_hat-after:', diff_z)
                            likelihood = np.sqrt(np.linalg.det(2*np.pi*S_t)) * math.exp(-0.5*((diff_z).T)@(np.linalg.inv(S_t))@(diff_z))
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
                        # if (mu_bar[0,0]-real_x) > 0.3 or (mu_bar[1,0]-real_y) > 0.3:
                            # flag = True
                        sigma_bar = (np.identity(sigma_bar.shape[0])-(K_t @ H_t)) @ sigma_bar

            else:
                ''' find correspondence by ICP '''
                print('>>>> find correspondence by ICP')
                icp_flag = True
                P = np.vstack((lm_x, lm_y))
                U = np.vstack((np_obs_x_m_noise, np_obs_y_m_noise)) #############(obs_lm_x, obs_lm_y)
                robot_xy = np.array([ [real_x],
                                      [real_y] ])
                cols = get_Rt_by_ICP( P,U, robot_xy, real_tehta )
        
                '''iterative update predict measurement'''
                for k in range(len(obs_lm_x)):
                    m_j_x = lm_x[cols[k]]
                    m_j_y = lm_y[cols[k]]
                    m_j_radi = lm_radi[cols[k]]
                                                    
                    '''get predict measurement'''
                    diff_x = m_j_x - bel_x
                    diff_y = m_j_y - bel_y
                    q = (diff_x ** 2) + (diff_y ** 2)
                    z_hat = np.array([ [np.sqrt(q)],
                                        [arctan2(diff_y, diff_x) - bel_theta],
                                        [m_j_radi] ])
                    print('z_hat: ',z_hat)           
                    H_t = np.array([ [-diff_x / np.sqrt(q), -diff_y / np.sqrt(q), 0],
                                        [diff_y / q, -diff_x / q, -1],
                                        [0,0,0] ])
                    S_t = (H_t @ sigma_bar @ (H_t.T)) + Q_t

                    '''save estimated measured data'''
                    np_z_hat=np.append(np_z_hat,z_hat)
                    
                    '''kalman gain and update belief'''
                    K_t = sigma_bar @ (H_t.T) @ np.linalg.inv(S_t)
                    print('K: ',K_t)
                    z_true = np.reshape(np_z_true[:,k], (-1,3))
                    z_true = z_true.T
                    # print('z_true-z_hat: ',z_true-z_hat)
                    # print(np_z_true[:,k])
                    print('mu_bar: ', mu_bar)
                    mu_bar = mu_bar+K_t@(z_true-z_hat)
                    # if (mu_bar[0,0]-real_x) > 0.3 or (mu_bar[1,0]-real_y) > 0.3:
                        # flag = True
                    print('mu_bar_after: ', mu_bar)
                    sigma_bar = (np.identity(sigma_bar.shape[0])-(K_t @ H_t)) @ sigma_bar
            np_z_hat = np.reshape(np_z_hat, (-1,3))
            np_z_hat = np_z_hat.T
            # print('np_z_hat: ', np_z_hat)

        '''update by GPS (here by ground truth)'''

        z_true = np.array([ [utm_x_loc_origin[0 , i]],
                            [utm_y_loc_origin[0 , i]],
                            [utm_t_loc_origin[0 , i]] ])
        print("KF-z_true: ", z_true)
        C = np.array([  [1,0,0],
                        [0,1,0],
                        [0,0,1] ])
        S_t = (C @ sigma_bar @ (C.T)) + R_t
        K_t = sigma_bar @ (C.T) @ np.linalg.inv(S_t)
        z_hat = C @ mu_bar
        print("KF-z_hat: ", z_hat)
        mu_bar = mu_bar+K_t@(z_true-z_hat)
        print("KF-(z_true-z_hat): ", z_true-z_hat)
        sigma_bar = (np.identity(sigma_bar.shape[0])-(K_t @ C)) @ sigma_bar

        '''update belief'''
        mu = mu_bar
        sigma = sigma_bar
        mu_x[0 , i] = mu[0 , 0]
        mu_y[0 , i] = mu[1 , 0]
        mu_theta[0 , i] = mu[2 , 0]
        obs_lm_number[0 , i] = len(np_z_hat)/3
        print("<<<<finish update: "+str(i))

        # if  flag==True: #len(obs_lm_x) != 0:
        #     # print("difference btw mu_bar and robot_xy_new: ", mu_bar, robot_xy_new)
        #     # tmp = (mu_bar[:2,0] - robot_xy_new[:,0])
        #     plot_traj((x_pos_true, y_pos_true, theta_pos_true), (mu_x, mu_y, mu_theta), (obs_lm_x, obs_lm_y, obs_lm_radi),(lm_x,lm_y,lm_radi),i, \
        #                 np_z_hat, np_z_true, (bel_x, bel_y, bel_theta), (real_x, real_y, real_tehta), cols, icp_flag, flag)   
        
        if i%10 == 0:
            plot_traj( (mu_x_hat, mu_y_hat, mu_theta_hat),(utm_x_loc_origin, utm_y_loc_origin, utm_t_loc_origin), (mu_x, mu_y, mu_theta), (np_obs_x_m_noise, np_obs_y_m_noise, obs_lm_radi),(lm_x,lm_y,lm_radi),i, \
                    np_z_hat, np_z_true, cols, icp_flag, flag) 
        # plot_traj(gps_states, belief_states, update_states, markers, markers_in_map, index, np_z_hat, np_z_true, cols, icp_flag, flag):
    # np.save('/home/ncslaber/class_material/EKF_localization_with_unknown_correspondences/data_ground_truth/mu_x', mu_x)
    # np.save('/home/ncslaber/class_material/EKF_localization_with_unknown_correspondences/data_ground_truth/mu_y', mu_y)
    # np.save('/home/ncslaber/class_material/EKF_localization_with_unknown_correspondences/data_ground_truth/mu_theta', mu_theta)
    # np.save('/home/ncslaber/class_material/EKF_localization_with_unknown_correspondences/data_ground_truth/obs_lm_number', obs_lm_number)
    # np.save('/home/ncslaber/class_material/EKF_localization_with_unknown_correspondences/data_ground_truth/obs_lm_vector', np_z_true)
    