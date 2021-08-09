# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 10:55:12 2021

@author: matteo
"""

from asynch_rl.nns.robot_net import ConvModel
from asynch_rl.envs.gymstyle_envs import DiscrGymStyleRobot

import os
import torch
import rospy
import rosnode
import tf2_ros as tf
import transformations
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from key_listener import RobotKeyListener


class RobotController:
    def __init__(self, target, namespace='robot', controller_frequency = 2.5, auto_initialize=True):
        self.ns = namespace
        self.controller_freq = controller_frequency
        self.controller_time_step = 1/self.controller_freq
        self.controller_time = 0
        
        self.params = None
        
        #Setting target
        self.setTargetLocation(target)
        self.key_listener = RobotKeyListener()
        
        if auto_initialize:
            self.initialize()
        
        self.nn_path = os.path.join( os.path.dirname(os.path.abspath(__file__)) ,"data","nn" )
        
        self.n_frames = 6
        self.n_actions = 9
        
        self.state_tensor_z1 = None

        
        
    def resetControllerTimer(self):
        self.controller_time = 0
    
    def setTargetLocation(self, target):
        #target must be a list containing x and y
        self.target = target
        
        
    
    def loadNet(self):
        self.net = ConvModel(partial_outputs = True, n_actions = self.n_actions, n_frames = self.n_frames, N_in = (135,4), \
                                  fc_layers = [40, 40, 20], softmax = True , layers_normalization = True,\
                                      n_partial_outputs = 18) 
            
        self.discr_gym_env = DiscrGymStyleRobot( n_frames = self.n_frames, n_bins_act= 2 )
        
        self.net.load_net_params(self.nn_path, "nn_policy" , torch.device('cpu') )
        
        
    def abs_2_rel(self, target_coordinates, current_pose):

        return self.discr_gym_env.env.robot.target_rel_position(target_coordinates = target_coordinates , current_pose = current_pose)

        
    def getRobotAction(self, state_obs ):
        
        action = torch.zeros([self.n_actions], dtype=torch.bool)
        net_input = self.discr_gym_env.get_net_input( state_obs, state_tensor_z1 = self.state_tensor_z1)
        
        prob_distrib, map_output = self.net(net_input, return_map = True)
        action_index = torch.argmax(prob_distrib)
        action[action_index]=True
        
        delta_v_x, delta_v_rot = self.discr_gym_env.boolarray_to_action(action.detach().numpy())
        
        self.state_tensor_z1 = net_input
        
        return delta_v_x, delta_v_rot
    
    
    def initialize(self):
        #Start the controller ROS node
        rospy.init_node('controller', anonymous=True)
        rospy.loginfo('Controller node successfully initialized')
        
        #self._loadROSParams()
        
        rospy.loginfo('Setting up tf buffer')
        self.tf_buffer = tf.Buffer()
        self.tf_listener = tf.TransformListener(self.tf_buffer)
        
        #Get static transform from base_link to laser_scanner
        self.scanner_static_tf = self.tf_buffer.lookup_transform('base_link','laserscanner_front_link',rospy.Time(),rospy.Duration(3))        
        rospy.loginfo('tf buffer initialized')
        
        self.last_pose = None
        self.current_pose = None
        self.last_scan = None
        self.current_scan = None
        self.enable = False
        
        #Initialize publishers
        rospy.loginfo('Initializing publishers')
        self.command_pub = rospy.Publisher("cmd_vel",Twist,queue_size=10)
        self.command_msg = Twist()
        self.emergency_command_msg = Twist()
        rospy.loginfo('Publishers Initialized')
        
        #Initialize subscribers
        rospy.loginfo('Initializing subscribers')
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self._odomCallback)
        self.scan_sub = rospy.Subscriber('/sick_s300_front/scan_filtered', LaserScan, self._scanCallback)
        rospy.loginfo('Subscribers Initialized')
        
        rospy.loginfo('Initializing control from keyboard')
        self.key_listener.initialize()
        self.key_listener.start()
        rospy.loginfo('Control from keyboard successfully initialized')
        
        
        rospy.loginfo('Systems Successfully initialized')
        
        
        
        
        
        
    def _processDataForNet(self):
        scan_data = np.array(self.current_scan.ranges)
        net_in_size = self.net.get_net_input_shape()[0][3]
        scan_data = self._decimateAndMinScanReadings(scan_data, net_in_size)
        robot_pose = [self.current_pose.pose.pose.position.x, self.current_pose.pose.pose.position.y]
        
        q = (self.current_pose.pose.pose.orientation.x,
             self.current_pose.pose.pose.orientation.y,
             self.current_pose.pose.pose.orientation.z,
             self.current_pose.pose.pose.orientation.w)
        euler = transformations.euler_from_quaternion(q)
        
        robot_orientation = euler[2]
        
        dist, angle = self.abs_2_rel(self.target, robot_pose + [robot_orientation])
        
        return (scan_data, np.array((self.current_pose.twist.twist.linear.x, self.current_pose.twist.twist.angular.z, dist, angle)))
        
        
       
        
    def _odomCallback(self,data):
        self.last_pose = self.current_pose
        self.current_pose = data
    
    def _scanCallback(self,data):
        self.last_scan = self.current_scan
        self.current_scan = data
        
    def _checkROSstatus(self):
        pass
    
    
    def run(self, print_debug_action=False):
        
        #Set loop frequency
        r = rospy.Rate(self.controller_freq)
        while not rospy.is_shutdown():
            if self.key_listener.stop_triggered or self.key_listener.emergency_stop_triggered:
                self.enable = False
            else:
                self.enable = True
                print('ciao')
                
            if self.enable:
                delta_vx, delta_dtheta = self.getRobotAction(self._processDataForNet())
                self.command_msg.linear.x = self.current_pose.twist.twist.linear.x + delta_vx/20
                self.command_msg.angular.z = self.current_pose.twist.twist.angular.z + delta_dtheta/20
                if print_debug_action:
                    rospy.loginfo('Commanded variation of linear speed:\t' + str(delta_vx))
                    rospy.loginfo('Commanded variation of angular speed:\t' + str(delta_dtheta))
                
                self.command_pub.publish(self.command_msg)
            else:
                self.command_pub.publish(self.emergency_command_msg)
                
            #Check if target is changed:
            if self.key_listener.new_target is not None:
                self.target = self.setTargetLocation(self.key_listener.new_target)
                self.key_listener.new_target = None
            
            if self.key_listener.homing_requested:
                self.target = self.setTargetLocation([0, 0])
                self.key_listener.homing_requested = False
            
                
            #self.command_pub.publish(self.command_msg)
            self.controller_time += self.controller_time_step
            
            r.sleep()
            
            
        if rospy.is_shutdown():
            rospy.loginfo('Shutting down the robot controller...')
            self.key_listener.stop()
            
            
            
    def _decimateAndMinScanReadings(self, original_scan, net_input_size):
        k = int(np.floor(original_scan.shape[0]/net_input_size))
        out_scan = np.zeros((net_input_size,))
        for i in range(out_scan.shape[0]):
            if i+k > len(original_scan):
                out_scan[i] = original_scan[i:].min()
            else:
                out_scan[i] = original_scan[i:i+k].min()
        
        return out_scan
        
        
    def _loadROSParams(self):
        rospy.loginfo('Attempting to load all ROS parameters for the controller')
        self.params = {'lidar_min_range':rospy.get_param("lidar_min_range", default=0.01),\
                       'lidar_max_range':rospy.get_param("lidar_max_range", default=15),\
                       'robot_max_lin_vel':rospy.get_param("robot_max_lin_vel", default=0.5),\
                       'robot_max_ang_vel':rospy.get_param("robot_max_ang_vel", default=0.6),\
                       'transform_scan_readings':rospy.get_param("transform_scan_readings", default=True),\
                       'debug_commanded_vels':rospy.get_param("debug_commanded_vels", default=True)}
        
                                                                    
        
    
    

if __name__ == "__main__":
    try:
        rob = RobotController([10,10])
        rob.loadNet()
        state = rob.discr_gym_env.env.reset()
        #print(state)
        rob.getRobotAction(state)
        print(rob.net.get_net_input_shape())
    #==============================================================================
    #     state = rob.discr_gym_env.env.reset()
    #     
    #     rob.getRobotAction(state)
    #         
    #     dist, ang = rob.abs_2_rel([20,10], [0,0, 0.785])
    #     print(f'dist = {dist}')
    #     print(f'ang = {ang}')    
    #==============================================================================
        rob.run(print_debug_action=True)
    except:
        rob.key_listener.stop()
    
    
    