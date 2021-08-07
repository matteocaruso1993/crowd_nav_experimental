# crowd_nav_experimental

## Setup
Dowload the project into a folder of your choice:

```git clone https://github.com/matteocaruso1993/crowd_nav_experimental.git $PATH-TO-YOUR-FOLDER$```

This package requires python=3.8 in order to be executed correctly. For this reason, it is needed to recompile the geometry pkg. First create an apposite catkin workspace. For Example:

```mkdir -p ~/catkin_test_ws/src```

```cd catkin_test_ws```

``catkin_make && cd src```

```git clone https://github.com/ros/geometry2.git && cp $PATH-TO-YOUR-FOLDER$/crowd_nav_robot_controller . && cd ..```

```catkin_make --cmake-args -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so```



