# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 13:20:21 2021

@author: matteo
"""

import os
import socket
import rospy


class RosManager:
    def __init__(self, master_uri=None):
        self.master_uri = os.getenv('ROS_MASTER_URI')
        
        
        
    def checkMasterURI(self, uri):
        if uri is not None:
            self.master_uri = uri
        elif uri is None and os.getenv('ROS_MASTER_URI') is not None:
            self.master_uri = os.getenv('ROS_MASTER_URI')
        else:
            rospy.logerr('Can not find a valid master URI. Please provide one...')
            
    
    def checkIfMasterRunning(self):
        try:
            master = rospy.get_master()
            master.getSystemState()
        except:
            print('Dioporco')
            
    def getCurrentIPAddresses(self):
        gw = os.popen("ip -4 route show default").read().split()
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        s.close()
        
        

            
    
    
                
            
        