#!/bin/bash

cd $HOME
sleep 5
source $HOME/.bashrc
sleep 0.1
#source $HOME/p38Env/bin/activate #Source python 3.8 virtual environment
source $HOME/catkin_python38/devel/setup.bash; #Source environment built with python 3
sleep 0.1
source $HOME/catkin_python38/devel/setup.bash;
sleep 0.1
source $HOME/catkin_python38/devel/setup.bash;
sleep 0.1
read -p "Start controller?"
roslaunch crowd_nav_robot_controller controller.launch simulated:=true 2>/dev/null

$SHELL