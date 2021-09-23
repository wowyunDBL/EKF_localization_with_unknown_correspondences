

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