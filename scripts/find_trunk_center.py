'''math tool'''
import csv
import numpy as np
from scipy.spatial import distance as dist

'''plot tool'''
import matplotlib.pyplot as plt

'''image tool'''
import cv2
import statistics as sta

import utm
from pyproj import Proj
import sys
if sys.platform.startswith('linux'): # or win
    print("in linux")
    file_path = "/home/ncslaber/110-1/210922_EKF-fusion-test/zigzag_bag/"


for AA in range(3627):
    ''' show raw data '''
    npDepth = np.load(file_path+"depth/"+str(AA)+".npy")
    npColor = np.load(file_path+"color/"+str(AA)+".npy")

    ''' to world coordinate '''
    cx_d = 320.6562194824219 #424
    cy_d = 241.57083129882812 #241
    fx_d = 384.31365966796875 #424
    fy_d = 384.31365966796875 #424
    npPointX = np.asarray(range(640))-cx_d
    npPointX = np.diag(npPointX)
    npPointX = npDepth.dot(npPointX)/ fx_d * (-1)

    npPointY = np.asarray(range(480))-cy_d
    npPointY = np.diag(npPointY)
    theta = 0/180*np.pi
    npPointY = npPointY.dot(npDepth)/ fy_d * (-1) 
    npPointY = npPointY*np.cos(theta) + npDepth * np.sin(theta) + 410
    npPointY = npPointY.astype('float16')
    ''' depth segmentation: show layers '''
    npHeight = np.copy(npPointY)
    ''' top-down view grid '''
    def depth_Z(u,v):
        return npDepth[v][u]
    height_layer_tmp = np.logical_and(npHeight<500,npHeight>300)
    height_layer = np.logical_and(height_layer_tmp,npHeight!=410)

    plane_l1 = np.zeros((int(10/0.05),int(10/0.05)),dtype=np.uint8)

    row, column = npDepth.shape

    for v in range(row):
        if height_layer[v].any() == True:
            for u in range(column):
                if height_layer[v][u] == True:
                    z_depth = depth_Z(u,v)
                    if (z_depth>50 and z_depth<6000):
                        x_depth = (u-cx_d)/fx_d*depth_Z(u,v)
                        plane_l1[200-int(z_depth/50)][int(x_depth/50)+100] += 1
                        
    hieght_or = np.zeros((int(10/0.05),int(10/0.05)), dtype=np.uint8)
    hieght_or[plane_l1>8]=255
    hieght_or = hieght_or.astype('uint8')
    kernel = np.ones((2,2), np.uint8)
    hieght_or = cv2.dilate(hieght_or,kernel,iterations = 1)
    hieght_or = cv2.erode(hieght_or, kernel, iterations = 1)

    ''' find connected component and push into point array A '''
    num_objects, labels = cv2.connectedComponents(hieght_or)
    centre_x_list = []
    centre_y_list = []
    radius_r_list = []
    circle_bd = np.zeros(hieght_or.shape, dtype=np.uint8)
    # print('>>>>num_objects:',num_objects)
    for i in range(num_objects-1):
        A = []
        for x in range(200):
            for y in range(200):
                if labels[x][y] == i+1:
                    A.append(np.array([-x/(x*x+y*y), -y/(x*x+y*y), -1/(x*x+y*y)]))
        A = np.asarray(A)
        # print('# of points: ',A.shape)
        if A.shape[0] < 10:
            continue

        k = np.linalg.inv(A.T @ A)
        k = k @ A.T
        k = k @ np.ones((k.shape[1],1))
        centre_x = k[0][0]/(-2)
        centre_y = k[1][0]/(-2)
        radius_r = np.sqrt(centre_x*centre_x+centre_y*centre_y-k[2][0])
        # print('x,y,r: ', int(centre_x+0.5), int(centre_y+0.5), int(radius_r+0.5))
        
        cv2.circle(circle_bd,(int(centre_y+0.5), int(centre_x+0.5)), int(radius_r+0.5), 150, 2)
        
        centre_x_list.append(int(centre_x+0.5))
        centre_y_list.append(int(centre_y+0.5))
        radius_r_list.append(int(radius_r+0.5))

        cv2.putText(circle_bd, #numpy array on which text is written
                    str(int(centre_x+0.5))+','+str(int(centre_y+0.5)), #text
                    (int(centre_y)-20,int(centre_x)-20), #position at which writing has to start
                    cv2.FONT_HERSHEY_SIMPLEX, #font family
                    0.2, #font size
                    255, #font color
                    1, cv2.LINE_AA) #font stroke
    circle_bd[hieght_or==255]=255
    centre_x_list = np.asarray(centre_x_list)
    centre_y_list = np.asarray(centre_y_list)
    radius_r_list = np.asarray(radius_r_list)

    ''' load robot current pose '''
    file_path = '/home/ncslaber/110-1/210922_EKF-fusion-test/zigzag_bag/'
    with open(file_path + 'cb_pose_lat_lon_theta.csv', 'r') as csvfile:
        robot_pose_gps = list( csv.reader(csvfile, delimiter=',') )
        robot_pose_gps = np.array(robot_pose_gps).astype(float)
    
    lat = robot_pose_gps[AA,0]
    # print(lat)
    lng = robot_pose_gps[AA,1]
    imu_yaw = robot_pose_gps[AA,2]
    
    _, _, zone, R = utm.from_latlon(lat, lng)
    proj = Proj(proj='utm', zone=zone, ellps='WGS84', preserve_units=False)
    utm_x_loc_origin, utm_y_loc_origin = proj(lng, lat)

    cX_m_loc = (centre_y_list-100)*0.05
    cY_m_loc = (200-centre_x_list)*0.05
    cX_utm_loc = cX_m_loc*np.cos(imu_yaw)-cY_m_loc*np.sin(imu_yaw) + utm_x_loc_origin
    cY_utm_loc = cX_m_loc*np.sin(imu_yaw)+cY_m_loc*np.cos(imu_yaw) + utm_y_loc_origin
    center_utm_loc = np.vstack((cX_utm_loc,cY_utm_loc))

    file_path = "/home/ncslaber/110-1/210922_EKF-fusion-test/zigzag_bag/"
    np.save(file_path+"found_center/"+str(AA)+'-x',cX_utm_loc )
    np.save(file_path+"found_center/"+str(AA)+'-y',cY_utm_loc )

    np.save(file_path+"found_center/"+str(AA)+'-q_x',cX_m_loc )
    np.save(file_path+"found_center/"+str(AA)+'-q_y',cY_m_loc )
    np.save(file_path+"found_center/"+str(AA)+'-r',radius_r_list )


fig3 = plt.figure(figsize=(8,8))
plt.title('current found trunk')
plt.imshow(cv2.cvtColor(circle_bd, cv2.COLOR_BGR2RGB))
plt.show()



