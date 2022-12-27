#!/bin/bash


source $HOME/p38Env/bin/activate    #Substitute here the path to your python 3.8 virtual environment


#Start crowd nav controller
file_path=$(rospack find crowd_nav_robot_controller)/bin/start_controller.sh


gnome-terminal -e "/bin/bash $file_path"



roslaunch crowd_nav_robot_controller start_full_sim.launch simulated:=true debug_connection:=false 2>/dev/null










