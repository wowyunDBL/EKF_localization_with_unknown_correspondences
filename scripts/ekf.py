# encoding: UTF-8
'''math tool'''
import csv
import numpy as np
from scipy.spatial import distance as dist

'''plot tool'''
import matplotlib.pyplot as plt
from matplotlib import animation

'''image tool'''
import cv2
import statistics as sta

import sys
# if sys.platform.startswith('linux'): # or win
#     print("in linux")
#     file_path = "/home/ncslaber/mapping_node/mapping_ws/src/mapping_explorer/0906_demo_data/2tree/"
#     # test_data/
# #     sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
# else:
#     file_path = r"C:/Users/15sin/OneDrive/文件/GitHub/mapping_explorer/test_data/"

def animate(true_states, belief_states, markers):
    x_tr, y_tr, th_tr = true_states
    x_guess, y_guess = belief_states
    
    radius = .5
    yellow = (1,1,0)
    black = 'k'
    world_bounds = [-10,10]
    
    fig = plt.figure()
    ax = plt.axes(xlim=world_bounds, ylim=world_bounds)
    ax.set_aspect('equal')
    ax.plot(markers[0], markers[1], '+', color=black)
    actual_path, = ax.plot([], [], color='b', zorder=-2, label="Actual")
    pred_path, = ax.plot([], [], color='r', zorder=-1, label="Predicted")
    heading, = ax.plot([], [], color=black)
    robot = plt.Circle((x_tr[0],y_tr[0]), radius=radius, color=yellow, ec=black)
    ax.add_artist(robot)
    ax.legend()

    def init():
        actual_path.set_data([], [])
        pred_path.set_data([], [])
        heading.set_data([], [])
        return actual_path, pred_path, heading, robot

    def animate(i):
        actual_path.set_data(x_tr[:i+1], y_tr[:i+1])
        pred_path.set_data(x_guess[:i+1], y_guess[:i+1])
        heading.set_data([x_tr[i], x_tr[i] + radius*cos(th_tr[i])], 
            [y_tr[i], y_tr[i] + radius*sin(th_tr[i])])
        robot.center = (x_tr[i],y_tr[i])
        return actual_path, pred_path, heading, robot
    save_mp4 = False
    if save_mp4 == False:
        anim = animation.FuncAnimation(fig, animate, init_func=init,
            frames=len(x_tr), interval=20, blit=True, repeat=False)
        plt.pause(.1)
        input("<Hit enter to close>")
    else:
        anim = animation.FuncAnimation(fig, animate, \
            np.arange(1,120), \
            interval=400,blit=False,init_func=init)
        anim.save('EKF_localization.mp4',fps=5)

if __name__ == "__main__":
    v_c = 1 + 0.5*cos(2*np.pi()*(0.2)*t)
    omg_c = -0.2 + 2*cos(2*np.pi()*(0.6)*t)
    alpha_1 = 0.1
    alpha_2 = .01
    alpha_3 = 0.01
    alpha_4 = .1