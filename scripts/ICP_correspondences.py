import numpy as np
from numpy import cos, sin, arctan2
from scipy.spatial import distance as dist

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