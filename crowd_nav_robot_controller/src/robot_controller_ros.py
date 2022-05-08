#!/usr/bin/env python3.8

# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 10:55:12 2021

@author: matteo
"""

#from asynch_rl.nns.robot_net import ConvModel
#from asynch_rl.envs.gymstyle_envs import DiscrGymStyleRobot

import os
from numpy.lib.polynomial import polyint
import torch
import rospy
import rosnode
import tf2_ros as tf
import transformations
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovariance, PointStamped, Point, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker
from key_listener import RobotKeyListener
from functions import clip, getClosestValue
import time
from copy import deepcopy


import traceback
import logging




class RobotController:
    def __init__(self, auto_initialize=True):
        self.way_point_idx = 0
        self.controller_freq = None
        self.controller_time_step = None
        self.controller_time = 0
        
        self.params = None
        self.n_frames = None
        self.n_actions = None
        
        self.state_tensor_z1 = None
        self.scan_topic_debug = None
        self.debug_scan_msg = None

        self.path_targets = list()
        self.path_targets_tr = list()
        
        #Setting target
        
        self.key_listener = RobotKeyListener()
        self.target_reached = False

        self.map2odom_transf = None
        
        if auto_initialize:
            self.initialize()
        
        self.setTargetLocation(self.path_targets[self.way_point_idx])
        
        self.nn_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)) ,"..","data","nn" ))

        
        
    def resetControllerTimer(self):
        self.controller_time = 0
    
    def setTargetLocation(self, target):
        #target must be a list containing x and y
        self.target = target


    def performInitialLocalization(self):
        rospy.loginfo("Running a test motion in order to perform self localization. Please wait...")
        tmp_msg = Twist()
        tic = time.time()

        #Go straight
        while True:
            if (time.time() - tic) <= 4:
                triggered = False
                tmp_msg.linear.x = 0.25
                tmp_msg.angular.z = 0
            else:
                triggered = True
                tmp_msg.linear.x = 0
                tmp_msg.angular.z = 0
                
            self.command_pub.publish(tmp_msg)
            if triggered:
                break

        tic = time.time()
        while True:
            if (time.time() - tic) <=12:
                triggered = False
                tmp_msg.angular.z = .25
            elif (time.time() - tic) >=12 and (time.time() - tic) <=36:
                triggered = False
                tmp_msg.angular.z = -.25
            elif (time.time() - tic) >=36 and (time.time() - tic) <=48:
                triggered = False
                tmp_msg.angular.z = -.25
            else:
                triggered = True
                tmp_msg.angular.z = 0

            self.command_pub.publish(tmp_msg)
            if triggered:
                break

        
        tic = time.time()
        #Go straight
        while True:
            if (time.time() - tic) <= 4:
                triggered = False
                tmp_msg.linear.x = -0.25
                tmp_msg.angular.z = 0
            else:
                triggered = True
                tmp_msg.linear.x = 0
                tmp_msg.angular.z = 0
                
            self.command_pub.publish(tmp_msg)
            if triggered:
                break



        rospy.loginfo("Test motion completed. Computed the initial transformation between the map and the odom frame")
        self.map2odom_transf = self.tf_buffer.lookup_transform("odom","map",rospy.Time(0))
        rospy.loginfo("The computed transformation is:")
        r = self.map2odom_transf.transform.rotation
        #print(self.map2odom_transf)
        q = (r.x,
             r.y,
             r.z,
             r.w)
        #print(transformations.euler_from_quaternion(q))

        
        
        #Turn right


        
        
    
    def loadNet(self):
        rospy.loginfo('Generating Conv Model')
        self.net = ConvModel(partial_outputs = True, n_actions = self.n_actions, n_frames = self.n_frames, N_in = (self.params['num_rays'],4), \
                                  fc_layers = [40, 40, 40], softmax = True , layers_normalization = True,\
                                      n_partial_outputs = 18)
        rospy.loginfo('Conv Model generated')
            
        self.discr_gym_env = DiscrGymStyleRobot(lidar_n_rays = self.params['num_rays'], n_frames = self.n_frames, n_bins_act= 4, v_linear_max=0.5, v_angular_max=1,max_v_x_delta=.2, max_v_rot_delta=.3)
        
        self.net.load_net_params(self.nn_path, "nn_policy_v3" , torch.device('cpu') )
        
        
    def abs_2_rel(self, target_coordinates, current_pose):

        return self.discr_gym_env.env.robot.target_rel_position(target_coordinates = target_coordinates , current_pose = current_pose)

        
    def getRobotAction(self, state_obs):
        action = torch.zeros([self.n_actions], dtype=torch.bool)
        net_input = self.discr_gym_env.get_net_input(state_obs, state_tensor_z1 = self.state_tensor_z1)
        prob_distrib, map_output = self.net(net_input, return_map = True)
        action_index = torch.argmax(prob_distrib)
        action[action_index]=True
        
        delta_v_x, delta_v_rot = self.discr_gym_env.boolarray_to_action(action.detach().numpy())
        
        self.state_tensor_z1 = net_input
        
        return delta_v_x, delta_v_rot
    
    
    def initialize(self):
        #Start the controller ROS node
        self.scan_data_initialized = False
        self.first_scan = None
        rospy.init_node('controller', anonymous=True)
        rospy.loginfo('Controller node successfully initialized')
        
        rospy.loginfo('Attempting to load all controller parameters')
        self._loadROSParams()
        rospy.loginfo('Parameters successfully loaded')
        
        rospy.loginfo('Setting up tf buffer')
        self.tf_buffer = tf.Buffer()
        self.tf_listener = tf.TransformListener(self.tf_buffer)
        
        #Get static transform from base_link to laser_scanner
        if not self.params['simulated']:
            self.scanner_static_tf = self.tf_buffer.lookup_transform(self.params['base_link_frame_name'],\
            self.params['scanner_link_frame_name'][0],rospy.Time(),rospy.Duration(3))
        else:
            self.scanner_static_tf = self.tf_buffer.lookup_transform(self.params['base_link_frame_name'],\
            self.params['scanner_link_frame_name'][1],rospy.Time(),rospy.Duration(3))
        #Duration added in order to being able to recive tf


        if self.params['listen_for_path_points']:
            self.points_listener = rospy.Subscriber("clicked_point", PointStamped, self._pointCallback)
            self.key_listener.new_path_enabled = True
            self.map2odom_transf = self.tf_buffer.lookup_transform("map",\
            "odom",rospy.Time(),rospy.Duration(30))
            #print(self.map2odom_transf)
        
        rospy.loginfo('tf buffer initialized')
        
        self.last_pose = None
        self.current_pose = None
        self.last_scan = None
        self.current_scan = None
        self.enable = False
        
        #Initialize publishers
        rospy.loginfo('Initializing publishers')
        self.command_pub = rospy.Publisher("cmd_vel",Twist,queue_size=10)
        if self.params['debug_scan']:
            self.scan_topic_debug = rospy.Publisher(self.params['debug_scanner_topic_name'], LaserScan, queue_size = 10)
            self.debug_scan_msg = LaserScan()
        
        self.command_msg = Twist()
        self.emergency_command_msg = Twist()


        self.path_pub = rospy.Publisher("path", Path, queue_size=10)
        self.path_msg = Path()

        rospy.loginfo('Publishers Initialized')


        if self.params['publish_markers']:
            self.marker_pub_original = rospy.Publisher('marker_map', Marker, queue_size=10)
            self.marker_pub_transformed = rospy.Publisher('marker_odom', Marker, queue_size=10)

            self.marker_msg = Marker()

        if self.params['publish_points']:
            self.point_pub_original = rospy.Publisher('point_map', PointStamped, queue_size=10)
            self.point_pub_transformed = rospy.Publisher('point_odom', PointStamped, queue_size=10)
            self.point_pub_target_odom = rospy.Publisher('target_odom', PointStamped, queue_size=10)

            self.point_msg = PointStamped()


        if self.params['display_target_location']:
            self.target_direction_pub = rospy.Publisher('target_direction', PoseStamped, queue_size=10)
            self.target_direction_msg = PoseStamped()
        
        #Initialize subscribers
        rospy.loginfo('Initializing subscribers')
        self.odom_sub = rospy.Subscriber('odom', Odometry, self._odomCallback)
        if not self.params['simulated']:
            self.scan_sub = rospy.Subscriber(self.params['scan_topic'][0], LaserScan, self._scanCallback)
        else:
            self.scan_sub = rospy.Subscriber(self.params['scan_topic'][1], LaserScan, self._scanCallback)
        
        self.amcl_sub = rospy.Subscriber('amcl_pose', PoseWithCovarianceStamped, self._amclCallback)
        rospy.loginfo('Subscribers Initialized')
        
        rospy.loginfo('Initializing control from keyboard')
        self.key_listener.initialize()
        self.key_listener.start()
        rospy.loginfo('Control from keyboard successfully initialized')
        
        
        rospy.loginfo('Systems Successfully initialized')
        
        
    def _printBanner(self):
        banner = """
        
░█████╗░██████╗░░█████╗░░██╗░░░░░░░██╗██████╗░███╗░░██╗░█████╗░██╗░░░██╗  ░░░░░░
██╔══██╗██╔══██╗██╔══██╗░██║░░██╗░░██║██╔══██╗████╗░██║██╔══██╗██║░░░██║  ░░░░░░
██║░░╚═╝██████╔╝██║░░██║░╚██╗████╗██╔╝██║░░██║██╔██╗██║███████║╚██╗░██╔╝  █████╗
██║░░██╗██╔══██╗██║░░██║░░████╔═████║░██║░░██║██║╚████║██╔══██║░╚████╔╝░  ╚════╝
╚█████╔╝██║░░██║╚█████╔╝░░╚██╔╝░╚██╔╝░██████╔╝██║░╚███║██║░░██║░░╚██╔╝░░  ░░░░░░
░╚════╝░╚═╝░░╚═╝░╚════╝░░░░╚═╝░░░╚═╝░░╚═════╝░╚═╝░░╚══╝╚═╝░░╚═╝░░░╚═╝░░░  ░░░░░░

██╗░░░██╗███╗░░██╗██╗████████╗░██████╗
██║░░░██║████╗░██║██║╚══██╔══╝██╔════╝
██║░░░██║██╔██╗██║██║░░░██║░░░╚█████╗░
██║░░░██║██║╚████║██║░░░██║░░░░╚═══██╗
╚██████╔╝██║░╚███║██║░░░██║░░░██████╔╝
░╚═════╝░╚═╝░░╚══╝╚═╝░░░╚═╝░░░╚═════╝░
"""
        print(banner)
        
        
        
        
        
        
    def _processDataForNet(self):
        try:
            self.setTargetLocation(self.path_targets_tr[self.way_point_idx])
        except:
            pass
        #print('Current Target Location:\t x=%4.4f,y=%4.4f' % tuple(self.target))
        #print('Current Target Index:\t x=%d' % self.way_point_idx)
        net_in_size = self.net.get_net_input_shape()[0][3]           
        if self.params['nearest_interp']:
            scan_data = np.zeros(self.reduced_scan_angles.shape)
            for i, ray in enumerate(self.reduced_scan_angles):
                scan_data[i] = self.current_scan.ranges[getClosestValue(self.scan_angles, ray, 'index')]
        else:
            scan_data = np.array(self.current_scan.ranges)
                
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
        #print('Target is located at %4.4f [m] with heading %4.4f [deg]' % (dist, np.rad2deg(angle)))
        
        if dist <= .5:
            self.way_point_idx += 1
            #print(self.target)
            #print(self.way_point_idx])
            
            if self.way_point_idx >= len(self.path_targets_tr):
                self.target_reached = True
            else:
                rospy.loginfo('Moving to the next assigned waypoint: [%4.4f,%4.4f]....' % (self.path_targets_tr[self.way_point_idx][0], self.path_targets_tr[self.way_point_idx][1]))

        
        if self.params['debug_scan']:
            self.debug_scan_msg.header.stamp = rospy.Time.now()
            self.debug_scan_msg.header.frame_id = "base_link"#self.params['base_link']
            self.debug_scan_msg.angle_min = self.reduced_scan_angles[0]
            self.debug_scan_msg.angle_max = self.reduced_scan_angles[-1]
            self.debug_scan_msg.angle_increment = self.reduced_scan_angles[1] - self.reduced_scan_angles[0]
            self.debug_scan_msg.range_min = self.params['lidar_min_range']
            self.debug_scan_msg.range_max = self.params['lidar_max_range']
            self.debug_scan_msg.ranges = tuple(scan_data)
            self.debug_scan_msg.intensities = tuple(len(scan_data)*[100])
            
            self.scan_topic_debug.publish(self.debug_scan_msg)
            
        #print(scan_data.shape)
        scan_data=scan_data[:-1]
        if not self.params['simulated']:
            scan_data = np.flipud(scan_data)
        
        if not self.params['normalize_observations']:
            return (scan_data, np.array((self.current_pose.twist.twist.linear.x, self.current_pose.twist.twist.angular.z, dist, angle)))
            
        else:
            return (scan_data/self.params['lidar_max_range'], np.array((
            self.current_pose.twist.twist.linear.x/self.params['robot_max_lin_vel'],\
            self.current_pose.twist.twist.angular.z/self.params['robot_max_lin_vel'], dist/self.params['target_distance_max'], angle/np.pi)))
        
        
       
        
    def _odomCallback(self,data):
        #rospy.loginfo('got 1')
        self.last_pose = self.current_pose
        self.current_pose = data
        
    
    def _scanCallback(self,data):
        #rospy.loginfo('got 2')
        self.last_scan = self.current_scan
        self.current_scan = data
        if not self.scan_data_initialized:
            self.scan_angles = np.arange(data.angle_min, data.angle_max, data.angle_increment)
            self.scan_angles.shape
            self.scan_data_initialized = True

    def _pointCallback(self, data):
        #data1 = self.tf_buffer.transform(data, "odom")
        l = [data.point.x, data.point.y]

        if not self.key_listener.catched:
            self.way_point_idx += 1
            self.key_listener.catched = True

        if self.params['debug_path_points']:
            print(l)

        if self.key_listener.path_requested:
            self.path_targets.append(l)

    def _amclCallback(self, data):
        pose = PoseStamped()
        pose.pose.position.x = data.pose.pose.position.x
        pose.pose.position.y = data.pose.pose.position.y
        self.path_msg.poses.append(pose)

    def publishPath(self):
        self.path_msg.header.frame_id = "map"
        self.path_msg.header.stamp = rospy.Time.now()
        self.path_pub.publish(self.path_msg)

            
        
    def _checkROSstatus(self):
        pass
    
    
    def run(self):
        #Set loop frequency
        r = rospy.Rate(self.controller_freq)
        while not rospy.is_shutdown():
            #print(self.target)
            try:
                self.map2odom_transf = self.tf_buffer.lookup_transform("map",\
                "odom",rospy.Time(),rospy.Duration(0))
                #print(self.map2odom_transf)
                #rospy.loginfo('Got TF')
            except:
                pass

            self.transform_targets()

            if self.params['publish_markers']:
                self.publishMarkersMsgs()

            if self.params['publish_points']:
                self.publishPointsMsgs()

            if self.params['display_target_location']:
                self.publishTargetLocation()
            
            self.publishPath()

            #Apply transform to current target
            
            if self.params['debug_scan']:            
                self._processDataForNet()
                
                
            if self.key_listener.stop_triggered or self.key_listener.emergency_stop_triggered or self.target_reached:
                self.enable = False
            else:
                self.enable = True

            if self.target_reached:
                rospy.loginfo('DESTINATION REACHED!')
                
            if self.enable:
                delta_vx, delta_dtheta = self.getRobotAction(self._processDataForNet())
                self.command_msg.linear.x = self.current_pose.twist.twist.linear.x + delta_vx/self.params['robot_lin_vel_scale']
                self.command_msg.angular.z = self.current_pose.twist.twist.angular.z + delta_dtheta/self.params['robot_ang_vel_scale']
                
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
                       'scan_topic':rospy.get_param("scanner_topic_name", default='sick_s300_front/scan_filtered'),\
                       'nearest_interp':rospy.get_param("nearest_interp", default=False),\
                       'scan_opening':rospy.get_param("scan_opening", default = None),\
                       'num_rays':rospy.get_param("num_rays", default = None),\
                       'debug_scan':rospy.get_param("debug_scan", default = False),\
                       'debug_scanner_topic_name':rospy.get_param("debug_scanner_topic_name", default = "scan_debug"),\
                       'simulated':rospy.get_param("simulated", False),\
                       'normalize_observations':rospy.get_param("normalize_observations", default = True),\
                       'target_distance_max':rospy.get_param("target_distance_max",default = 20),\
                       'listen_for_path_points':rospy.get_param("listen_for_path_points", default= True),\
                       'save_path_points':rospy.get_param("save_path_points", default=True),\
                       'debug_path_points':rospy.get_param("debug_path_points", default = False),\
                       'publish_markers':rospy.get_param("publish_markers", default=True),\
                       'publish_points':rospy.get_param("publish_points", default=True),\
                       'display_target_location':rospy.get_param("display_target_location", default = True)}
                       
        self.controller_freq = float(self.params['controller_frequency'])
        self.controller_time_step = 1/self.controller_freq
        self.n_frames = int(self.params['n_frames'])
        self.n_actions = int(self.params['n_actions'])
        if self.params['nearest_interp']:
            if self.params['scan_opening'] is None or self.params['num_rays'] is None:
                rospy.logerr('Invalid data for scan_opening or num_rays paramters. Using default settings...')
                self.params['nearest_interp'] = False
            else:
                self.reduced_scan_angles = np.linspace(-np.deg2rad(self.params['scan_opening'])/2, np.deg2rad(self.params['scan_opening'])/2,self.params['num_rays'])
                #print(self.reduced_scan_angles.shape)

        self.path_targets.append(self.params['target'])
        self.path_targets_tr = deepcopy(self.path_targets)
        
                                                                    
    def _printLegend(self):
        print(80*'=')
        print('\t\t\t\t LEGEND')
        print(80*'=')
        print('*\tPress SPACE key to activate/deactivate temporary stop')
        print('*\tPress S key to activate emergency stop. To remove it, it will be requested to press the ENTER key')
        print('*\tPress N key to insert a new target location')
        print('*\tPress H key to make the robot return to its homing position')
        print('*\tPress P key to to pass waypoints through Rviz')
        print('*\tPress L key to to load waypoints from file')
        print(80*'=')

    def transform_targets(self):
        self.path_targets_tr = deepcopy(self.path_targets)
        r = self.map2odom_transf.transform.rotation
        #print(self.map2odom_transf)
        q = (r.x,
             r.y,
             r.z,
             r.w)
        theta = transformations.euler_from_quaternion(q)[2]
        #print(transformations.euler_from_quaternion(q))
        pt1 = [0,0]
        for i, pt in enumerate(self.path_targets):
            pt1[0] = pt[0] - self.map2odom_transf.transform.translation.x
            pt1[1] = pt[1] - self.map2odom_transf.transform.translation.y



            self.path_targets_tr[i][0] = pt1[0]*np.cos(theta) + pt1[1]*np.sin(theta)
            self.path_targets_tr[i][1] = -pt1[0]*np.sin(theta) + pt1[1]*np.cos(theta)

            #self.path_targets_tr[i][0] -= self.map2odom_transf.transform.translation.x
            #self.path_targets_tr[i][1] -= self.map2odom_transf.transform.translation.y


        #print(self.path_targets_tr)
        #print(len(self.path_targets))
        #print(self.path_targets)


    def publishMarkersMsgs(self):
        #Original
        self.marker_msg.header.stamp = rospy.Time.now()
        p = Point()
        p.x = self.path_targets[self.way_point_idx][0]
        p.y = self.path_targets[self.way_point_idx][1]

        self.marker_msg.header.frame_id = "map"
        self.marker_msg.type = self.marker_msg.POINTS
        self.marker_msg.action = self.marker_msg.ADD

        # self.marker_msg.pose.position.x = self.path_targets[self.way_point_idx][0]
        # self.marker_msg.pose.position.y = self.path_targets[self.way_point_idx][1]
        self.marker_msg.pose.orientation.w = 1
        self.marker_msg.scale.x = 0.4
        self.marker_msg.scale.y = 0.4
        self.marker_msg.scale.z = 0.4
        self.marker_msg.color.a = 1
        self.marker_msg.color.r = 1
        self.marker_msg.points = [p]
        self.marker_pub_original.publish(self.marker_msg)


        p.x = self.path_targets_tr[self.way_point_idx][0]
        p.y = self.path_targets_tr[self.way_point_idx][1]
        self.marker_msg.header.stamp = rospy.Time.now()
        self.marker_msg.header.frame_id = "odom"
        self.marker_msg.type = self.marker_msg.POINTS
        self.marker_msg.action = self.marker_msg.ADD
        self.marker_msg.pose.orientation.w = 1
        self.marker_msg.scale.x = 0.4
        self.marker_msg.scale.y = 0.4
        self.marker_msg.scale.z = 0.4
        self.marker_msg.color.a = 0.5
        self.marker_msg.color.r = 0.5
        self.marker_msg.points = [p]
        self.marker_pub_transformed.publish(self.marker_msg)


    def publishPointsMsgs(self):
        self.point_msg.header.stamp = rospy.Time.now()
        self.point_msg.header.frame_id = "map"

        self.point_msg.point.x = self.path_targets[self.way_point_idx][0]
        self.point_msg.point.y = self.path_targets[self.way_point_idx][1]
        self.point_pub_original.publish(self.point_msg)


        self.point_msg.header.stamp = rospy.Time.now()
        self.point_msg.header.frame_id = "odom"
        self.point_msg.point.x = self.path_targets_tr[self.way_point_idx][0]
        self.point_msg.point.y = self.path_targets_tr[self.way_point_idx][1]
        self.point_pub_transformed.publish(self.point_msg)


        self.point_msg.header.stamp = rospy.Time.now()
        self.point_msg.header.frame_id = "odom"
        self.point_msg.point.x = self.target[0]
        self.point_msg.point.y = self.target[1]
        self.point_pub_target_odom.publish(self.point_msg)


    def publishTargetLocation(self):

        robot_pose = [self.current_pose.pose.pose.position.x, self.current_pose.pose.pose.position.y]
        q = (self.current_pose.pose.pose.orientation.x,
             self.current_pose.pose.pose.orientation.y,
             self.current_pose.pose.pose.orientation.z,
             self.current_pose.pose.pose.orientation.w)
        euler = transformations.euler_from_quaternion(q)
        
        robot_orientation = euler[2]
        dist, angle = self.abs_2_rel(self.target, robot_pose + [robot_orientation])

        tot_angle = robot_orientation + angle
        self.target_direction_msg.header.stamp = rospy.Time.now()
        self.target_direction_msg.header.frame_id = "odom"
        self.target_direction_msg.pose.position.x = robot_pose[0]
        self.target_direction_msg.pose.position.y = robot_pose[1]

        q = transformations.quaternion_from_euler(0,0,tot_angle)
        self.target_direction_msg.pose.orientation.x = q[0]
        self.target_direction_msg.pose.orientation.y = q[1]
        self.target_direction_msg.pose.orientation.z = q[2]
        self.target_direction_msg.pose.orientation.w = q[3]
        self.target_direction_pub.publish(self.target_direction_msg)



    
    

if __name__ == "__main__":
    #try:
        rob = RobotController()
        rob.loadNet()
        #rospy.loginfo('Net successfully loaded')
        state = rob.discr_gym_env.env.reset()
        state = (state[0][:-1],state[1])
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
        rob._printBanner()
        rob._printLegend()
        #rob.performInitialLocalization()
        #try:
        rob.run()
        #except e:
        #    print(e)

        #    pass
    #except Exception as e:
    #    rospy.loginfo('Shutting down robot controller')
    #    rob.key_listener.stop()
    #    logging.error(traceback.format_exc())
    #    print(e)
    
    
    
