'''math tool'''
import csv
import math
import numpy as np
from numpy import cos, sin, arctan2
from scipy.spatial import distance as dist

'''plot tool'''
import matplotlib.pyplot as plt
from matplotlib import animation

# def plot_traj(belief_states, gps_states, update_states, markers, markers_in_map, index, np_z_hat, np_z_true, cols, icp_flag, flag):
#     x_tr, y_tr, th_tr = gps_states
#     x_init_guess, y_init_guess, theta_init_guess = belief_states   
#     x_update, y_update, theta_update = update_states

#     # x_tr_lm, y_tr_lm, th_tr_lm = lm_states

#     radius = 0.5
    
#     # world_bounds = [-15,10]
#     fig, ax = plt.subplots(figsize=(10,10))
#     # ax = plt.axes(xlim=world_bounds, ylim=world_bounds)
#     ax.set_aspect('equal')

#     '''plot landmarkers'''
#     plt.scatter(markers_in_map[0], markers_in_map[1], marker='X',s=100, color='g', label='ref landmarks')
#     plt.scatter(markers[0], markers[1], marker='X',s=100, color='r', label='gps_obs landmarks')
#     number_of_point=12
#     piece_rad = np.pi/(number_of_point/2)
#     for j in range( len(markers_in_map[0]) ):
#         neg_bd = []
#         for i in range(number_of_point):
#             neg_bd.append((markers_in_map[0][j]+markers_in_map[2][j]*np.cos(piece_rad*i), markers_in_map[1][j]+markers_in_map[2][j]*np.sin(piece_rad*i)))
#         neg_bd=np.asarray(neg_bd)
#         plt.scatter(neg_bd[:,0], neg_bd[:,1], c='k', s=10)
#     for j in range( len(markers[0]) ):
#         neg_bd = []
#         for i in range(number_of_point):
#             neg_bd.append((markers[0][j]+markers[2][j]*np.cos(piece_rad*i), markers[1][j]+markers[2][j]*np.sin(piece_rad*i)))
#         neg_bd=np.asarray(neg_bd)
#         plt.scatter(neg_bd[:,0], neg_bd[:,1], c='k', s=10)

#     '''plot traj'''
#     plt.scatter(x_tr[0][:index], y_tr[0][:index], color='b', label="gps_states", s=10)
#     # plt.scatter(x_init_guess, y_init_guess, color='b', label="gps_states", s=10)
#     plt.scatter(x_update[0][:index], y_update[0][:index], color='g', label="update_states", s=10)
          
#     '''plot final state'''
#     plt.scatter(x_tr[0][index],y_tr[0][index], s=300, color='lightblue', ec='k', label='GPS pose')
#     plt.plot( [x_tr[0][index], x_tr[0][index] + radius*cos(th_tr[0][index]) ], 
#                 [y_tr[0][index], y_tr[0][index] + radius*sin(th_tr[0][index]) ], color='k' )
#     plt.scatter(x_update[0][index],y_update[0][index], s=500, color='y', ec='k', label='Predicted pose')
#     plt.plot( [x_update[0][index], x_update[0][index] + radius*cos(theta_update[0][index]) ], 
#                 [y_update[0][index], y_update[0][index] + radius*sin(theta_update[0][index]) ], color='k' )
#     # plt.scatter(x_tr_lm[0][index],y_tr_lm[0][index], s=400, color='lightblue', ec='k', label='landmark pose')
#     # plt.plot( [x_tr_lm[0][index], x_tr_lm[0][index] + radius*cos(th_tr_lm[0][index]) ], 
#     #           [y_tr_lm[0][index], y_tr_lm[0][index] + radius*sin(th_tr_lm[0][index]) ], color='k' )

#     '''plot observation z'''
#     plot_measured_landmarks(np_z_hat, np_z_true, (x_init_guess[0][index], y_init_guess[0][index], theta_init_guess[0][index]), \
#                                                     (x_tr[0][index], y_tr[0][index], th_tr[0][index]), \
#                                                         (x_update[0][index], y_update[0][index], theta_update[0][index]))
#     if flag == True:
#         if icp_flag == True:
#             plt.text(3.0, 8, 'icp matched points: '+str(cols),fontsize=14)
#         else:
#             plt.text(3.0, 8, 'maximum matched point',fontsize=14)

#     '''plot obs lm
#     for i in range( len(markers[0]) ):
#         plt.plot([ markers[0][i],x_tr[0][index] ],
#                 [ markers[1][i],y_tr[0][index] ], color='k')'''
#     plt.scatter(x_update[0][:index], y_update[0][:index], color='r', label="Predicted", s=10)
#     plt.title('update times: '+str(index)+'/200', fontsize=25)
#     plt.yticks(fontsize=20)
#     plt.xticks(fontsize=20)
#     plt.legend(fontsize=15)
#     plt.show()
#     # fig.savefig('/home/ncslaber/110-1/210922_EKF-fusion-test/zigzag_bag/TMP'+str(index)+'.png')

def plot_traj(true_states, belief_states, markers, markers_in_map, index, np_z_hat, np_z_true, bel_pose, real_pose, cols, icp_flag):
    x_tr, y_tr, th_tr = true_states
    x_guess, y_guess, theta_guess = belief_states
    # x_tr_lm, y_tr_lm, th_tr_lm = lm_states

    radius = 0.5
    
    world_bounds = [-15,10]
    fig, ax = plt.subplots(figsize=(10,10),dpi=120)
    ax = plt.axes(xlim=world_bounds, ylim=world_bounds)
    ax.set_aspect('equal')

    '''plot landmarkers'''
    plt.scatter(markers_in_map[0], markers_in_map[1], marker='X',s=100, color='g', label='ref landmarks')
    plt.scatter(markers[0], markers[1], marker='X',s=100, color='r', label='real_obs landmarks')
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
    
      
    '''plot final state'''
    plt.scatter(x_tr[0][index],y_tr[0][index], s=300, color='lightblue', ec='k', label='Actual pose')
    plt.plot( [x_tr[0][index], x_tr[0][index] + radius*cos(th_tr[0][index]) ], 
               [y_tr[0][index], y_tr[0][index] + radius*sin(th_tr[0][index]) ], color='k' )
    plt.scatter(x_guess[0][index],y_guess[0][index], s=500, color='y', ec='k', label='Predicted pose')
    plt.plot( [x_guess[0][index], x_guess[0][index] + radius*cos(theta_guess[0][index]) ], 
              [y_guess[0][index], y_guess[0][index] + radius*sin(theta_guess[0][index]) ], color='k' )
    # plt.scatter(x_tr_lm[0][index],y_tr_lm[0][index], s=400, color='lightblue', ec='k', label='landmark pose')
    # plt.plot( [x_tr_lm[0][index], x_tr_lm[0][index] + radius*cos(th_tr_lm[0][index]) ], 
    #           [y_tr_lm[0][index], y_tr_lm[0][index] + radius*sin(th_tr_lm[0][index]) ], color='k' )

    '''plot observation z'''
    plot_measured_landmarks(np_z_hat, np_z_true, bel_pose, real_pose)
    if icp_flag == True:
        plt.text(3.0, 8, 'icp matched points: '+str(cols),fontsize=14)
    else:
        plt.text(3.0, 8, 'maximum matched point',fontsize=14)

    '''plot obs lm
    for i in range( len(markers[0]) ):
        plt.plot([ markers[0][i],x_tr[0][index] ],
                [ markers[1][i],y_tr[0][index] ], color='k')'''
    plt.scatter(x_guess[0][:index+1], y_guess[0][:index+1], color='r', label="Predicted", s=10)
    plt.title('update times: '+str(index)+'/200', fontsize=25)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.legend(fontsize=15)
    plt.show()
    # fig.savefig('/home/ncslaber/class_material/EKF_localization_with_unknown_correspondences/images/sample_5_times/'+str(index)+'.png')

def plot_measured_landmarks(np_z_hat, np_z_true, bel_pose, real_pose):
    bel_x, bel_y, bel_theta = bel_pose
    real_x, real_y, real_theta = real_pose
    radius = 0.5

    # world_bounds = [-15,10]
    # fig, ax = plt.subplots(figsize=(10,10),dpi=120)
    # ax = plt.axes(xlim=world_bounds, ylim=world_bounds)
    # ax.set_aspect('equal')

    '''plot state'''
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
        wr_x =  cos(real_theta)*r_x - sin(real_theta)*r_y
        wr_x += real_x
        wr_y = sin(real_theta)*r_x + cos(real_theta)*r_y
        wr_y += real_y
        plt.plot([ real_x,wr_x ],
                 [ real_y,wr_y ], color='b', linestyle='-', label='real_observing')

# def plot_measured_landmarks(np_z_hat, np_z_true, bel_pose, gps_pose, update_pose):
#     bel_x, bel_y, bel_theta = bel_pose
#     gps_x, gps_y, gps_theta = gps_pose
#     update_x, update_y, update_theta = update_pose
#     radius = 0.5

#     # world_bounds = [-15,10]
#     # fig, ax = plt.subplots(figsize=(10,10),dpi=120)
#     # ax = plt.axes(xlim=world_bounds, ylim=world_bounds)
#     # ax.set_aspect('equal')

#     '''plot state'''
#     # bel_theta -= np.pi/2
#     # gps_theta -= np.pi/2
#     plt.scatter(bel_x, bel_y, s=300, color='lightyellow', ec='k', label='z_hat pose')
#     plt.plot( [bel_x, bel_x + radius*cos(bel_theta) ], 
#                [bel_y, bel_y + radius*sin(bel_theta) ], color='k' )
    
#     '''plot observation'''
    
#     for i in range( np_z_true.shape[1] ):
#         r_x = np_z_hat[0][i] * cos(np_z_hat[1][i])
#         r_y = np_z_hat[0][i] * sin(np_z_hat[1][i])
#         wr_x =  cos(bel_theta)*r_x - sin(bel_theta)*r_y
#         wr_x += bel_x
#         wr_y = sin(bel_theta)*r_x + cos(bel_theta)*r_y
#         wr_y += bel_y
#         plt.plot([ bel_x,wr_x ],
#                  [ bel_y,wr_y ], color='k', linestyle='--', label='predi_observing')

#         r_x = np_z_true[0][i] * cos(np_z_true[1][i])
#         r_y = np_z_true[0][i] * sin(np_z_true[1][i])
#         wr_x =  cos(gps_theta)*r_x - sin(gps_theta)*r_y
#         wr_x += gps_x
#         wr_y = sin(gps_theta)*r_x + cos(gps_theta)*r_y
#         wr_y += gps_y
#         plt.plot([ gps_x,wr_x ],
#                  [ gps_y,wr_y ], color='b', linestyle='-', label='gps_observing')

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
    
