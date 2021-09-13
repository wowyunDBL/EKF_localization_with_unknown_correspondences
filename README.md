# EKF_localization_with_unknown_correspondences

## Compile it 
```sh
mkdir build
cd build
cmake ..
make
```
## Eigen
[Eigen](http://eigen.tuxfamily.org/dox-devel/group__QuickRefPage.html)
* Only need to include header file: `#include <Eigen/Dense>`
* Remember to add the following to your CMakeLists.txt:
```c++
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})
```

## Reference
[Problem set](https://github.com/adlarkin/Probabilistic_robotics/tree/master/EKF)
[Code modified form](https://github.com/nghiaho12/EKF_localization_known_correspondences)