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
    def __init__(self, namespace='robot'):
        self.ns = namespace
        
        self.nn_path = os.path.join( os.path.dirname(os.path.abspath(__file__)) ,"nn" )
        
        self.n_frames = 6
        self.n_actions = 9
        
        self.state_tensor_z1 = None
    
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
        print('To-DO')
        
        
    def _checkROSstatus(self):
        pass
    
    

if __name__ == "__main__":
    
    rob = RobotController()
    rob.loadNet()
        
    state = rob.discr_gym_env.env.reset()
    
    rob.getRobotAction(state)
        
    dist, ang = rob.abs_2_rel([20,10], [0,0, 0.785])
    print(f'dist = {dist}')
    print(f'ang = {ang}')    
        
    
    
    