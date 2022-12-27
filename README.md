# crowd_nav_experimental
## Introduction
This ROS package implements a controller for a differential drive mobile robot, which implements a neural network which has been trained with A3C Reinforcement Learning. The package allows the interaction with the visualization software Rviz, and allows the user to communicate waypoints directly from the keyboard.
The package can be either be used in the simulation software Gazebo, or with a real differential drive robot. Moreover, the package can be either installed on a remote computer, or on the onboard computer of the robot.

## Setup

### Install requirements
Python3 requirement:

```
cd catkin_test_ws/src
python3.6 -m pip install empy
sudo apt-get install ros-melodic-teleop-twist*
sudo apt-get install ros-melodic-teleop-tools*
sudo apt-get install ros-melodic-mouse-teleop
sudo apt-get install ros-melodic-gmapping*
sudo apt-get install python-catkin-tools
sudo apt-get install ros-melodic-map-server
sudo apt-get install ros-melodic-amcl*
```



Dowload the project into a folder of your choice:

```git clone https://github.com/matteocaruso1993/crowd_nav_experimental.git $PATH-TO-YOUR-FOLDER$```

This package requires python==3.8 in order to be executed correctly. For this reason, it is needed to recompile the geometry pkg. First create an apposite catkin workspace. For Example:

```mkdir -p ~/catkin_test_ws/src```

```cd catkin_test_ws```

```catkin_make && cd src```

```git clone https://github.com/ros/geometry2.git && cp $PATH-TO-YOUR-FOLDER$/crowd_nav_robot_controller . && cd ..```

```catkin_make --cmake-args -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so```


Activate python 3.8 virtual environment and install specific requirements:
```
source p38Env/bin/activate
cd src/crowd_nav_experimental
pip install -r requirements.txt
```

Install now the repositories needed for crowd nav simulations

* [robot_sf](https://github.com/matteocaruso1993/robot-sf)
```
mkdir -p $HOME/Documents/repositories/python
cd $HOME/Documents/repositories/python
git clone https://github.com/matteocaruso1993/robot-sf.git
cd robot-sf
pip install -r requirements.txt
pip install -e .
cd ..
```

* [asynch_rl](https://github.com/matteocaruso1993/asynch-rl)
```
git clone https://github.com/matteocaruso1993/asynch-rl.git
cd asynch-rl
pip install -r requirements.txt
pip install -e .
cd 
```



Download now the neobotix needed packages:

* [neo_simulation](https://github.com/neobotix/neo_simulation)
```
cd $HOME/catkin_ws_test/src
git clone https://github.com/neobotix/neo_simulation.git
cd neo_simulation
git checkout melodic
cd ..
```
If no errors, you can proceed with the next steps.

## Test
In this section, follows the instructions in order to 

To start the controller then:

```source devel/setup.bash```

```roslaunch roslaunch crowd_nav_robot_controller controller.launch```

If no errors then the package is working correctly.

N.B. Make sure that the required topic names matches correctly, otherwise it is needed to modify the .yaml configuration file.





