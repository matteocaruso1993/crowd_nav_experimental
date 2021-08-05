# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 10:55:12 2021

@author: matteo
"""

from asynch_rl.nns.robot_net import ConvModel
from asynch_rl.envs.gymstyle_envs import DiscrGymStyleRobot

import os
import torch
#import rospy
#import rosnode
#import tf
#from sensor_msgs.msg import LaserScan
#from nav_msgs.msg import Odometry


class RobotController:
    def __init__(self, target, namespace='robot', controller_frequency = 2.5, auto_initialize=True):
        self.ns = namespace
        self.controller_freq = controller_frequency
        self.controller_time_step = 1/self.controller_freq
        self.controller_time = 0
        
        #Setting target
        self.setTargetLocation(target)
        
        if auto_initialize:
            self.initialize()
        
        self.nn_path = os.path.join( os.path.dirname(os.path.abspath(__file__)) ,"nn" )
        
        self.n_frames = 6
        self.n_actions = 9
        
        self.state_tensor_z1 = None
        
        
    def resetControllerTimer(self):
        self.controller_time = 0
    
    def setTargetLocation(self, target):
        #target must be a list containing x and y
        self.target = target
        
    
    def loadNet(self):
        self.net = ConvModel(partial_outputs = True, n_actions = self.n_actions, n_frames = self.n_frames, N_in = (268,4), \
                                  fc_layers = [40, 40, 20], softmax = True , layers_normalization = True,\
                                      n_partial_outputs = 18) 
            
        self.discr_gym_env = DiscrGymStyleRobot( n_frames = self.n_frames, n_bins_act= 2 )
        
        self.net.load_net_params(self.nn_path, "nn_policy" , torch.device('cpu') )
        
        
    def abs_2_rel(self, target_coordinates, current_pose):

        return self.discr_gym_env.env.robot.target_rel_position( target_coordinates = target_coordinates , current_pose = current_pose)

        
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
        #Initialize subscription and publishing to topics
    
        #Initialize tf listener
        self.tf_listener = tf.listener()
        self.scanner_static_tf = self.tf_listener.lookupTransform('/laserscanner_front_link','base_link')
        
                
        #Start the controller ROS node
        rospy.init_node('/controller', anonymous=True)
        self.last_pose = None
        self.current_pose = None
        self.last_scan = None
        self.current_scan = None
        self.enable = False
        
        #Initialize publishers
        self.command_pub = rospy.Publisher('/cmd_vel',Twist,queue_size=10)
        self.command_msg = Twist()
        
        #Initialize subscribers
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self._odomCallback)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self._scanCallback)
        
        
    def _processDataForNet(self):
        scan_data = np.array(self.current_scan.ranges)
        robot_pose = [self.current_pose.pose.position.x, self.current_pose.pose.position.y]
        
        q = (self.current_pose.pose.orientation.x,
             self.current_pose.pose.orientation.y,
             self.current_pose.pose.orientation.z,
             self.current_pose.pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(q)
        
        robot_orientation = euler[2]
        
        
        
        
        
        print('TO DO')
        
        
    def _odomCallback(self,data):
        self.last_pose = self.current_pose
        self.current_pose = data
    
    def _scanCallback(self,data):
        self.last_scan = self.current_scan
        self.current_scan = data
        
        
    def _checkROSstatus(self):
        pass
    
    
    def run(self):
        r = rospy.Rate(self.controller_freq)
        while not rospy.is_shutdown():
            #DO things
            #self.controller
            continue
            rospy.sleep()
            
            
        if rospy.is_shutdown():
            rospy.loginfo('Shutting down the robot controller...')
    
    

if __name__ == "__main__":
    
    rob = RobotController()
    rob.loadNet()
        
    state = rob.discr_gym_env.env.reset()
    
    rob.getRobotAction(state)
        
    dist, ang = rob.abs_2_rel([20,10], [0,0, 0.785])
    print(f'dist = {dist}')
    print(f'ang = {ang}')    
        
    
    
    