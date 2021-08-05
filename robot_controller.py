# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 10:55:12 2021

@author: matteo
"""

import rospy
import rosnode
import tf
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry


class RobotController:
    def __init__(self, namespace='robot'):
        self.ns = namespace
    
    def loadNet(self, net):
        self.net = net
    
    def initialize(self):
        #Initialize subscription and publishing to topics
    
        #Initialize tf listener
        print('To-DO')
        
        
    def _checkROSstatus(self):
        
        
        
    
    
    