import csv
import numpy as np
from numpy import cos, sin, arctan2
import utm
from pyproj import Proj

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
            continue
        else: 
            lm_x.append(center[0])
            lm_y.append(center[1])
            lm_radi.append(center[2])
            print('load landmark map successfully with size = ', lm_x.shape)
    # center_utm_ref = np.load(file_path_map+'center_utm_ref.npy')
    # lm_x = [-7,2,3,5,6,6,3] 
    # lm_y = [8,7,6,5,4,-4,-6] 
    # lm_radi = [0.2, 0.5, 0.3, 0.2, 0.5, 0.3, 0.4]
    
    
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

def get_filtered_map_pose(file_path):
    
    with open(file_path + 'cb_pose.csv', 'r') as csvfile:
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