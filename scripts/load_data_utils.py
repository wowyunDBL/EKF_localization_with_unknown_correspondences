import csv
import numpy as np
from numpy import cos, sin, arctan2
import utm
from pyproj import Proj

def get_landmark():
        
    lm_x = [-7,2,3,5,6,6,3] 
    lm_y = [8,7,6,5,4,-4,-6] 
    lm_radi = [0.2, 0.5, 0.3, 0.2, 0.5, 0.3, 0.4]
        
    return lm_x, lm_y, lm_radi

def get_landmark_from_file():
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
            continue
        else: 
            lm_x.append(center[0])
            lm_y.append(center[1])
            lm_radi.append(center[2])
            print('load landmark map successfully with size = ', len(lm_x))

    return lm_x, lm_y, lm_radi

def get_observed_lm(file_path, index):
    # file_path = "/home/ncslaber/110-1/210922_EKF-fusion-test/zigzag_bag/"
    # obs_lm_x = []
    # obs_lm_y = []
    # obs_lm_radi = []
    np_z_true = []
    obs_lm_utm_x = np.load(file_path+"found_center/"+str(index)+'-x.npy')
    obs_lm_utm_y = np.load(file_path+"found_center/"+str(index)+'-y.npy')
    obs_lm_x = np.load(file_path+"found_center/"+str(index)+'-q_x.npy')
    obs_lm_y = np.load(file_path+"found_center/"+str(index)+'-q_y.npy')
    obs_lm_radi = np.load(file_path+"found_center/"+str(index)+'-r.npy')
    obs_lm_radi = obs_lm_radi/ 10  # should be 20
    for c in range( len(obs_lm_x) ):
        diff_x = obs_lm_x[c] 
        diff_y = obs_lm_y[c] 
        diff_theta = arctan2(diff_y, diff_x)

        q = (diff_x ** 2) + (diff_y ** 2)
        z_true = np.array([ [np.sqrt(q)],
                            [diff_theta],
                            [obs_lm_radi[c]] ])

        np_z_true=np.append(np_z_true, z_true)

    np_z_true = np.reshape(np_z_true, (-1,3))
    np_z_true = np_z_true.T    

    return np_z_true, obs_lm_utm_x, obs_lm_utm_y, obs_lm_radi

def get_observed_lm_for_sim(mu_bar, global_lm):
    obs_lm_x = []
    obs_lm_y = []
    obs_lm_radi = []
    FOV = np.pi/2
    for i in range( len(global_lm[0]) ):
        diff_x = global_lm[0][i] - mu_bar[0]
        diff_y = global_lm[1][i] - mu_bar[1]
        diff_theta = arctan2(diff_y, diff_x) - mu_bar[2]

        if diff_theta > -FOV and diff_theta < FOV:
            if diff_x*diff_x+diff_y*diff_y < 64:
                obs_lm_x.append( global_lm[0][i])
                obs_lm_y.append( global_lm[1][i])
                obs_lm_radi.append( global_lm[2][i])
                
    return obs_lm_x, obs_lm_y, obs_lm_radi

def get_filtered_map_pose(file_path):
    
    with open(file_path + 'cb_pose.csv', 'r') as csvfile:
        robot_pose_gps = list( csv.reader(csvfile, delimiter=',') )
        robot_pose_gps = np.array(robot_pose_gps).astype(float)
    print("load filtered_map_pose size = ", robot_pose_gps.shape)
    '''project from gps to utm
    lat = robot_pose_gps[:,0]
    lng = robot_pose_gps[:,1]
    _, _, zone, R = utm.from_latlon(lat, lng)
    proj = Proj(proj='utm', zone=zone, ellps='WGS84', preserve_units=False)
    filtered_map_x, filtered_map_y = proj(lng, lat)
    filtered_map_x.shape = (1,-1)
    filtered_map_y.shape = (1,-1)'''

    filtered_map_x = robot_pose_gps[:,0]
    filtered_map_y = robot_pose_gps[:,1]
    filtered_map_x.shape = (1,-1)
    filtered_map_y.shape = (1,-1)
    '''load yaw (0 is north)'''
    filtered_map_t = robot_pose_gps[:,2]
    filtered_map_t.shape = (1,-1)

    return filtered_map_x, filtered_map_y, filtered_map_t

def get_GPS(file_path):
    
    with open(file_path + 'cb_pose_lat_lon_theta.csv', 'r') as csvfile:
        robot_pose_gps = list( csv.reader(csvfile, delimiter=',') )
        robot_pose_gps = np.array(robot_pose_gps).astype(float)
    print("load filtered_map_pose size = ", robot_pose_gps.shape)
    '''project from gps to utm'''
    lat = robot_pose_gps[:,0]
    lng = robot_pose_gps[:,1]
    _, _, zone, R = utm.from_latlon(lat, lng)
    proj = Proj(proj='utm', zone=zone, ellps='WGS84', preserve_units=False)
    filtered_map_x, filtered_map_y = proj(lng, lat)
    filtered_map_x.shape = (1,-1)
    filtered_map_y.shape = (1,-1)
    
    '''load yaw (0 is north)'''
    filtered_map_t = robot_pose_gps[:,2]
    filtered_map_t.shape = (1,-1)

    return filtered_map_x, filtered_map_y, filtered_map_t


def generate_ground_truth(t):

    x_pos_true = np.zeros(t.shape)
    y_pos_true = np.zeros(t.shape)
    theta_pos_true = np.zeros(t.shape)

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
    np.save('/home/ncslaber/class_material/EKF_localization_with_unknown_correspondences/data_ground_truth/theta_pos_true', theta_pos_true)

    return x_pos_true, y_pos_true, theta_pos_true