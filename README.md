# EKF_localization_with_unknown_correspondences

## Compile it 
```sh
mkdir build
cd build
cmake ..
make
```

## tree
```bash
├── scripts
│   ├── EKF_localization.py  # module
│   ├── ekf_with_icp_unknown.py  # sim
│   ├── ekf_with_KF_GPS.py  # sim
│   ├── ekf_with_lm_and_odom.py  # sim
│   ├── ekf_with_lm_and_odom_realData.py
│   ├── ekf_with_mixLikelihood.py  # sim
│   ├── find_trunk_center.py
│   ├── ICP_correspondences.py  # module
│   ├── load_data_utils.py  # module
│   ├── plot_utils.py  # module
└── src
    └── EKF_localization.h

.
├── cb_pose.csv  #/outdoor_waypoint_nav/odometry/filtered
├── cb_pose_filter_map.csv #/outdoor_waypoint_nav/odometry/filtered_map
├── cb_pose_lat_lon_theta.csv #/outdoor_waypoint_nav/gps/filtered
├── color/  # saved bgr npy
├── depth/
├── found_center/ # trunk center of utm
├── found_center_filtered_map/
├── index_timestamp.csv
├── index_timestamp_filter_map.csv
└── TMP
```

## Usage
#### find correspondence by ICP
```bash
python ekf_with_icp_unknown.py
```
plz save the trajectory in ./data_ground_truth and ./data_predicted

## Eigen
[Eigen](http://eigen.tuxfamily.org/dox-devel/group__QuickRefPage.html)
* Only need to include header file: `#include <Eigen/Dense>`
* Remember to add the following to your CMakeLists.txt:
```c++
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})
```

## Reference
[Problem spec](https://github.com/adlarkin/Probabilistic_robotics/tree/master/EKF)  
[Code modified from](https://github.com/nghiaho12/EKF_localization_known_correspondences)