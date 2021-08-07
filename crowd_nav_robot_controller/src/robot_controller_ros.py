#!/usr/bin/env python3.8

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
from functions import clip


class RobotController:
    def __init__(self, auto_initialize=True):
        self.controller_freq = None
        self.controller_time_step = None
        self.controller_time = 0
        
        self.params = None
        self.n_frames = None
        self.n_actions = None
        
        self.state_tensor_z1 = None
        
        #Setting target
        
        self.key_listener = RobotKeyListener()
        
        if auto_initialize:
            self.initialize()
        
        self.setTargetLocation(self.params['target'])
        
        self.nn_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)) ,"..","data","nn" ))
        

        
        
    def resetControllerTimer(self):
        self.controller_time = 0
    
    def setTargetLocation(self, target):
        #target must be a list containing x and y
        self.target = target
        
        
    
    def loadNet(self):
        rospy.loginfo('Generating Conv Model')
        self.net = ConvModel(partial_outputs = True, n_actions = self.n_actions, n_frames = self.n_frames, N_in = (268,4), \
                                  fc_layers = [40, 40, 20], softmax = True , layers_normalization = True,\
                                      n_partial_outputs = 18)
        rospy.loginfo('Conv Model generated')
            
        self.discr_gym_env = DiscrGymStyleRobot(n_frames = self.n_frames, n_bins_act= 2 )
        
        self.net.load_net_params(self.nn_path, "nn_policy" , torch.device('cpu') )
        
        
    def abs_2_rel(self, target_coordinates, current_pose):

        return self.discr_gym_env.env.robot.target_rel_position(target_coordinates = target_coordinates , current_pose = current_pose)

        
    def getRobotAction(self, state_obs):
        
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
        
        rospy.loginfo('Attempting to load all controller parameters')
        self._loadROSParams()
        rospy.loginfo('Parameters successfully loaded')
        
        rospy.loginfo('Setting up tf buffer')
        self.tf_buffer = tf.Buffer()
        self.tf_listener = tf.TransformListener(self.tf_buffer)
        
        #Get static transform from base_link to laser_scanner
        self.scanner_static_tf = self.tf_buffer.lookup_transform(self.params['base_link_frame_name'],\
        self.params['scanner_link_frame_name'],rospy.Time(),rospy.Duration(3))
        #Duration added in order to being able to recive tf
        
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
        self.odom_sub = rospy.Subscriber('odom', Odometry, self._odomCallback)
        self.scan_sub = rospy.Subscriber(self.params['scan_topic'], LaserScan, self._scanCallback)
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
        
        #Clip scanner readings
        if self.params['clip_readings']:
            scan_data[scan_data <= self.params['lidar_min_range']] = self.params['lidar_max_range']
            scan_data[scan_data >= self.params['lidar_max_range']] = self.params['lidar_max_range']
            
        
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
        #rospy.loginfo('got 1')
        self.last_pose = self.current_pose
        self.current_pose = data
    
    def _scanCallback(self,data):
        #rospy.loginfo('got 2')
        self.last_scan = self.current_scan
        self.current_scan = data
        
    def _checkROSstatus(self):
        pass
    
    
    def run(self):
        #Set loop frequency
        r = rospy.Rate(self.controller_freq)
        while not rospy.is_shutdown():
            if self.key_listener.stop_triggered or self.key_listener.emergency_stop_triggered:
                self.enable = False
            else:
                self.enable = True
                
            if self.enable:
                delta_vx, delta_dtheta = self.getRobotAction(self._processDataForNet())
                self.command_msg.linear.x = (self.current_pose.twist.twist.linear.x + delta_vx)/self.params['robot_lin_vel_scale']
                self.command_msg.angular.z = (self.current_pose.twist.twist.angular.z + delta_dtheta)/self.params['robot_ang_vel_scale']
                
                if self.params['clip_speed']:
                    if self.params['exclude_negative_speed']:
                        min_vel = 0
                    else:
                        min_vel = -self.params['robot_max_lin_vel']
                    
                    max_vel = self.params['robot_max_lin_vel']
                    self.command_msg.linear.x = clip(self.command_msg.linear.x, min_vel, max_vel)
                    self.command_msg.angular.z = clip(self.command_msg.angular.z, -self.params['robot_max_ang_vel'], self.params['robot_max_ang_vel'])
                    
                
                if self.params['debug_commanded_vels']:
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
                       'debug_commanded_vels':rospy.get_param("debug_commanded_vels", default=True),\
                       'controller_frequency':rospy.get_param("controller_frequency", default=2.5),\
                       'base_link_frame_name':rospy.get_param("base_link_name",default="base_link"),\
                       'scanner_link_frame_name':rospy.get_param("scanner_link_name",default="laserscanner_front_link"),\
                       'n_frames':rospy.get_param("n_frames",default=6),\
                       'n_actions':rospy.get_param("n_actions",default=9),\
                       'clip_speed':rospy.get_param("clip_speed", default=True),\
                       'clip_readings':rospy.get_param("clip_readings", default=True),\
                       'robot_lin_vel_scale':rospy.get_param("scale_commanded_lin_vel", default=1),\
                       'robot_ang_vel_scale':rospy.get_param("scale_commanded_ang_vel", default=1),\
                       'exclude_negative_speed':rospy.get_param("exclude_negative_speed",default=False),\
                       'target':rospy.get_param("target",default=[10,10]),\
                       'scan_topic':rospy.get_param("scanner_topic_name", default='sick_s300_front/scan_filtered')}
                       
        self.controller_freq = float(self.params['controller_frequency'])
        self.controller_time_step = 1/self.controller_freq
        self.n_frames = int(self.params['n_frames'])
        self.n_actions = int(self.params['n_actions'])
        
        
                                                                    
    def _printLegend(self):
        print(80*'=')
        print('\t\t\t\t LEGEND')
        print(80*'=')
        print('*\tPress SPACE key to activate/deactivate temporary stop')
        print('*\tPress S key to activate emergency stop. To remove it, it will be requested to press the ENTER key')
        print('*\tPress N key to insert a new target location')
        print('*\tPress H key to make the robot return to its homing position')
        print(80*'=')
    
    

if __name__ == "__main__":
    try:
        rob = RobotController()
        rob.loadNet()
        #rospy.loginfo('Net successfully loaded')
        state = rob.discr_gym_env.env.reset()
        #print(state)
        rob.getRobotAction(state)
        #rospy.loginfo(rob.nn_path)
    #==============================================================================
    #     state = rob.discr_gym_env.env.reset()
    #     
    #     rob.getRobotAction(state)
    #         
    #     dist, ang = rob.abs_2_rel([20,10], [0,0, 0.785])
    #     print(f'dist = {dist}')
    #     print(f'ang = {ang}')    
    #==============================================================================
        rospy.loginfo('SYSTEM READY: Starting controller loop. Press CTRL+C to terminate it.')
        rob._printLegend()
        rob.run()
    except:
        rospy.loginfo('Shutting down robot controller')
        rob.key_listener.stop()
    
    
    
